#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import sys

import chainer
import numpy as np
import random
import six
import torch
import torch.nn.functional as F

from chainer import reporter
from torch.autograd import Variable

from ctc_prefix_score import CTCPrefixScore
from e2e_asr_common import end_detect
from e2e_asr_common import label_smoothing_dist

from e2e_asr_attctc_th import AttAdd
from e2e_asr_attctc_th import AttCov
from e2e_asr_attctc_th import AttCovLoc
from e2e_asr_attctc_th import AttDot
from e2e_asr_attctc_th import AttLoc
from e2e_asr_attctc_th import AttLoc2D
from e2e_asr_attctc_th import AttLocRec
from e2e_asr_attctc_th import AttMultiHeadAdd
from e2e_asr_attctc_th import AttMultiHeadDot
from e2e_asr_attctc_th import AttMultiHeadLoc
from e2e_asr_attctc_th import AttMultiHeadMultiResLoc
from e2e_asr_attctc_th import CTC
from e2e_asr_attctc_th import Decoder as ASRDecoder
from e2e_asr_attctc_th import Encoder
from e2e_asr_attctc_th import NoAtt

from e2e_asr_attctc_th import to_cuda
from e2e_asr_attctc_th import lecun_normal_init_parameters
from e2e_asr_attctc_th import pad_list
from e2e_asr_attctc_th import set_forget_bias_to_one
from e2e_asr_attctc_th import mask_by_length
from e2e_asr_attctc_th import th_accuracy


torch_is_old = torch.__version__.startswith("0.3.")

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


class Reporter(chainer.Chain):
    def report(self, loss_st, loss_st_sub, loss_asr, loss_asr_sub, loss_mt,
               acc_st, acc_st_sub, acc_asr, acc_asr_sub, acc_mt, mtl_loss):
        reporter.report({'loss_st': loss_st}, self)
        reporter.report({'loss_st_sub': loss_st_sub}, self)
        reporter.report({'loss_asr': loss_asr}, self)
        reporter.report({'loss_asr_sub': loss_asr_sub}, self)
        reporter.report({'loss_mt': loss_mt}, self)

        reporter.report({'acc': acc_st}, self)
        reporter.report({'acc_st_sub': acc_st_sub}, self)
        reporter.report({'acc_asr': acc_asr}, self)
        reporter.report({'acc_asr_sub': acc_asr_sub}, self)
        reporter.report({'acc_mt': acc_mt}, self)

        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    def __init__(self, predictor, st_sub, asr, asr_sub, mt):
        super(Loss, self).__init__()
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

        # for MTL
        self.st = 1 - st_sub - asr - asr_sub - mt
        self.st_sub = st_sub
        self.asr = asr
        self.asr_sub = asr_sub
        self.mt = mt

    def forward(self, x):
        '''Loss forward

        :param x:
        :return:
        '''
        self.loss = None
        loss_st, loss_st_sub, loss_asr, loss_asr_sub, loss_mt, acc_st, acc_st_sub, acc_asr, acc_asr_sub, acc_mt = self.predictor(
            x)

        self.loss = 0.

        if self.st > 0:
            self.loss += self.st * loss_st
            loss_st_data = loss_st.data[0] if torch_is_old else float(loss_st)
        else:
            loss_st_data = None
        if self.st_sub > 0:
            self.loss += self.st_sub * loss_st_sub
            loss_st_sub_data = loss_st_sub.data[0] if torch_is_old else float(loss_st_sub)
        else:
            loss_st_sub_data = None

        if self.asr > 0:
            self.loss += self.asr * loss_asr
            loss_asr_data = loss_asr.data[0] if torch_is_old else float(loss_asr)
        else:
            loss_asr_data = None

        if self.asr_sub > 0:
            self.loss += self.asr_sub * loss_asr_sub
            loss_asr_sub_data = loss_asr_sub.data[0] if torch_is_old else float(loss_asr_sub)
        else:
            loss_asr_sub_data = None

        if self.mt > 0:
            self.loss += self.mt * loss_mt
            loss_mt_data = loss_mt.data[0] if torch_is_old else float(loss_mt)
        else:
            loss_mt_data = None

        loss_data = self.loss.data[0] if torch_is_old else float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_st_data, loss_st_sub_data,
                                 loss_asr_data, loss_asr_sub_data, loss_mt_data,
                                 acc_st, acc_st_sub, acc_asr, acc_asr_sub, acc_mt, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)

        return self.loss


class E2E(torch.nn.Module):
    def __init__(self, idim, odim, args, asr_model=None):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir

        # for MTL
        self.st = 1 - args.asr - args.asr_sub - args.mt - args.st_sub
        self.st_sub = args.st_sub
        self.asr = args.asr
        self.asr_type = args.asr_type
        self.asr_sub = args.asr_sub
        self.asr_type_sub = args.asr_type_sub
        self.mt = args.mt

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        if self.st > 0:
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                               self.subsample, args.dropout_rate)
        # for ASR
        if args.asr > 0:
            if args.asr_type == 'ctc':
                self.ctc_asr = CTC(odim, args.eprojs, args.dropout_rate)
            elif args.asr_type == 'att':
                self.att_asr = AttLoc(args.eprojs, args.dunits,
                                      args.adim, args.aconv_chans, args.aconv_filts)
                self.dec_asr = ASRDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                                          self.sos, self.eos, self.att_asr, self.verbose, self.char_list,
                                          labeldist, args.lsm_weight, args.sampling_probability)
        if args.asr_sub > 0:
            if args.asr_type_sub == 'ctc':
                self.ctc_asr_sub = CTC(odim, args.eprojs, args.dropout_rate)
            elif args.asr_type_sub == 'att':
                self.att_asr_sub = AttLoc(args.eprojs, args.dunits,
                                          args.adim, args.aconv_chans, args.aconv_filts)
                self.dec_asr_sub = ASRDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                                              self.sos, self.eos, self.att_asr_sub, self.verbose, self.char_list,
                                              labeldist, args.lsm_weight, args.sampling_probability)
        # for MT
        if args.mt > 0:
            self.embed_mt = torch.nn.Embedding(odim, args.dunits)
            self.enc_mt = Encoder('blstmp', args.dunits, 2, args.eunits, args.eprojs,
                                  [1, 1, 1], args.dropout_rate)
            self.att_mt = AttDot(args.eprojs, args.dunits, args.adim)
            self.dec_mt = MTDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                                    self.sos, self.eos, self.att_mt, self.verbose, self.char_list,
                                    labeldist, args.lsm_weight, args.sampling_probability,
                                    args.input_feeding)
        # for ST
        if self.st > 0:
            self.att_st = AttDot(args.eprojs, args.dunits, args.adim)
            self.dec_st = MTDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                                    self.sos, self.eos, self.att_st, self.verbose, self.char_list,
                                    labeldist, args.lsm_weight, args.sampling_probability,
                                    args.input_feeding)
        if args.st_sub > 0:
            self.att_st_sub = AttDot(args.eprojs, args.dunits, args.adim)
            self.dec_st_sub = MTDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                                        self.sos, self.eos, self.att_st_sub, self.verbose, self.char_list,
                                        labeldist, args.lsm_weight, args.sampling_probability,
                                        args.input_feeding)

        # weight initialization
        self.init_like_chainer()
        # additional forget-bias init in encoder ?
        # for m in self.modules():
        #     if isinstance(m, torch.nn.LSTM):
        #         for name, p in m.named_parameters():
        #             if "bias_ih" in name:
        #                 set_forget_bias_to_one(p)

        # transfer learning from pre-trained models
        self.init_pretrained_model(asr_model)

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)

        if self.st > 0:
            # exceptions
            # embed weight ~ Normal(0, 1)
            self.dec_st.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in six.moves.range(len(self.dec_st.decoder)):
                set_forget_bias_to_one(self.dec_st.decoder[l].bias_ih)
        if self.st_sub > 0:
            # exceptions
            # embed weight ~ Normal(0, 1)
            self.dec_st_sub.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in six.moves.range(len(self.dec_st_sub.decoder)):
                set_forget_bias_to_one(self.dec_st_sub.decoder[l].bias_ih)

        if self.asr > 0 and self.asr_type == 'att':
            # exceptions
            # embed weight ~ Normal(0, 1)
            self.dec_asr.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in six.moves.range(len(self.dec_asr.decoder)):
                set_forget_bias_to_one(self.dec_asr.decoder[l].bias_ih)
        if self.asr_sub > 0 and self.asr_type_sub == 'att':
            # exceptions
            # embed weight ~ Normal(0, 1)
            self.dec_asr_sub.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in six.moves.range(len(self.dec_asr_sub.decoder)):
                set_forget_bias_to_one(self.dec_asr_sub.decoder[l].bias_ih)

        if self.mt > 0:
            # exceptions
            # embed weight ~ Normal(0, 1)
            self.dec_mt.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in six.moves.range(len(self.dec_mt.decoder)):
                set_forget_bias_to_one(self.dec_mt.decoder[l].bias_ih)

    def init_pretrained_model(self, asr=None, mt=None, rnnlm=None):
        # Initialize encoder with pre-trained ASR
        if asr is not None:
            logging.info('Initialize the encoder with pre-trained ASR')
            asr_enc_param_dict = dict(asr.predictor.enc.named_parameters())
            for n, p in self.enc.named_parameters():
                p.data = asr_enc_param_dict[n].data

        # Initialize decoder with pre-trained MT
        if mt is not None:
            logging.info('Initialize the decoder with pre-trained MT')
            raise NotImplementedError

    # x[i]: ('utt_id', {'ilen':'xxx',...}})
    def forward(self, data):
        '''E2E forward

        :param data:
        :return:
        '''
        # utt list of frame x dim
        xs = [d[1]['feat'] for d in data]
        # remove 0-output-length utterances
        tids_tgt = [d[1]['output'][0]['tokenid'].split() for d in data]
        if self.asr > 0 or self.mt > 0:
            tids_src = [d[1]['output'][1]['tokenid'].split() for d in data]
        if self.st_sub > 0:
            tids_tgt_sub = [d[1]['output'][2]['tokenid'].split() for d in data]
        if self.asr_sub > 0:
            tids_src_sub = [d[1]['output'][3]['tokenid'].split() for d in data]
        filtered_index = filter(lambda i: len(tids_tgt[i]) > 0, range(len(xs)))
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
        if len(sorted_index) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(sorted_index)))
        xs = [xs[i] for i in sorted_index]
        # utt list of olen
        if self.st > 0 or self.mt > 0:
            ys_tgt = [np.fromiter(map(int, tids_tgt[i]), dtype=np.int64)
                      for i in sorted_index]
            if torch_is_old:
                ys_tgt = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training)) for y in ys_tgt]
            else:
                ys_tgt = [to_cuda(self, torch.from_numpy(y)) for y in ys_tgt]

        if self.asr > 0:
            ys_src_asr = [np.fromiter(map(int, tids_src[i]), dtype=np.int64)
                          for i in sorted_index]
            if torch_is_old:
                ys_src_asr = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training))
                              for y in ys_src_asr]
            else:
                ys_src_asr = [to_cuda(self, torch.from_numpy(y)) for y in ys_src_asr]

        if self.mt > 0:
            filtered_index_mt = filter(lambda i: len(tids_tgt[i]) > 0, range(len(tids_src)))
            sorted_index_mt = sorted(filtered_index_mt, key=lambda i: -len(tids_src[i]))
            if len(sorted_index_mt) != len(tids_src):
                logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                    len(tids_src), len(sorted_index_mt)))

            ys_src_mt = [np.fromiter(map(int, tids_src[i]), dtype=np.int64)
                         for i in sorted_index_mt]
            if torch_is_old:
                ys_src_mt = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training))
                             for y in ys_src_mt]
            else:
                ys_src_mt = [to_cuda(self, torch.from_numpy(y)) for y in ys_src_mt]

            olens = np.fromiter((len(xx) + 2 for xx in tids_src), dtype=np.int64)
            olens = [olens[i] for i in sorted_index_mt]

            # append <SOS> and <EOS>
            eos = Variable(ys_src_mt[0].data.new([self.eos]))
            sos = Variable(ys_src_mt[0].data.new([self.sos]))
            ys_src_mt = [torch.cat([sos, y, eos], dim=0) for y in ys_src_mt]
            ys_src_mt_pad = pad_list(ys_src_mt, self.eos)

        if self.st_sub > 0:
            ys_tgt_sub = [np.fromiter(map(int, tids_tgt_sub[i]), dtype=np.int64)
                          for i in sorted_index]
            if torch_is_old:
                ys_tgt_sub = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training))
                              for y in ys_tgt_sub]
            else:
                ys_tgt_sub = [to_cuda(self, torch.from_numpy(y)) for y in ys_tgt_sub]

        if self.asr_sub > 0:
            ys_src_asr_sub = [np.fromiter(map(int, tids_src_sub[i]), dtype=np.int64)
                              for i in sorted_index]
            if torch_is_old:
                ys_src_asr_sub = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training))
                                  for y in ys_src_asr_sub]
            else:
                ys_src_asr_sub = [to_cuda(self, torch.from_numpy(y)) for y in ys_src_asr_sub]

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        if torch_is_old:
            hs = [to_cuda(self, Variable(torch.from_numpy(xx), volatile=not self.training)) for xx in xs]
        else:
            hs = [to_cuda(self, torch.from_numpy(xx)) for xx in xs]

        # 1. encoder
        if self.st > 0:
            xpad = pad_list(hs, 0.0)
            hpad, hlens = self.enc(xpad, ilens)

        # 2. ST loss
        if self.st > 0:
            loss_st, acc_st = self.dec_st(hpad, hlens, ys_tgt)
            loss_st = loss_st.unsqueeze(0)
        else:
            loss_st, acc_st = None, None
        if self.st_sub > 0:
            loss_st_sub, acc_st_sub = self.dec_st_sub(hpad, hlens, ys_tgt_sub)
            loss_st_sub = loss_st_sub.unsqueeze(0)
        else:
            loss_st_sub, acc_st_sub = None, None

        # 3. ASR loss
        if self.asr > 0:
            if self.asr_type == 'ctc':
                loss_asr = self.ctc_asr(hpad, hlens, ys_src_asr)
                acc_asr = None
            elif self.asr_type == 'att':
                loss_asr, acc_asr = self.dec_asr(hpad, hlens, ys_src_asr)
                loss_asr = loss_asr.unsqueeze(0)
        else:
            loss_asr, acc_asr = None, None

        if self.asr_sub > 0:
            if self.asr_type_sub == 'ctc':
                loss_asr_sub = self.ctc_asr_sub(hpad, hlens, ys_src_asr_sub)
                acc_asr_sub = None
            elif self.asr_type_sub == 'att':
                loss_asr_sub, acc_asr_sub = self.dec_asr_sub(hpad, hlens, ys_src_asr_sub)
                loss_asr_sub = loss_asr_sub.unsqueeze(0)
        else:
            loss_asr_sub, acc_asr_sub = None, None

        # 4. MT loss
        if self.mt > 0:
            eys_mt = self.embed_mt(ys_src_mt_pad)
            hpad_mt, hlens_mt = self.enc_mt(eys_mt, olens)
            loss_mt, acc_mt = self.dec_mt(hpad_mt, hlens_mt, ys_tgt)
            loss_mt = loss_mt.unsqueeze(0)
        else:
            loss_mt, acc_mt = None, None

        return loss_st, loss_st_sub, loss_asr, loss_asr_sub, loss_mt, acc_st, acc_st_sub, acc_asr, acc_asr_sub, acc_mt

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        if torch_is_old:
            h = to_cuda(self, Variable(torch.from_numpy(
                np.array(x, dtype=np.float32)), volatile=True))
        else:
            h = to_cuda(self, torch.from_numpy(
                np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h = h.contiguous()
        h, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(h).data[0]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = self.dec_st.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)
        # TODO(hirofumi): remoe lpz

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, data):
        '''E2E attention calculation

        :param list data: list of dicts of the input (B)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
         :rtype: float ndarray
        '''
        if not torch_is_old:
            torch.set_grad_enabled(False)

        # utt list of frame x dim
        xs = [d[1]['feat'] for d in data]

        # remove 0-output-length utterances
        tids_tgt = [d[1]['output'][0]['tokenid'].split() for d in data]
        if self.st == 0 and self.mt > 0:
            tids_src = [d[1]['output'][1]['tokenid'].split() for d in data]
        filtered_index = filter(lambda i: len(tids_tgt[i]) > 0, range(len(xs)))
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
        if len(sorted_index) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(sorted_index)))
        xs = [xs[i] for i in sorted_index]

        # utt list of olen
        ys_tgt = [np.fromiter(map(int, tids_tgt[i]), dtype=np.int64)
                  for i in sorted_index]
        if torch_is_old:
            ys_tgt = [to_cuda(self, Variable(torch.from_numpy(y), volatile=True)) for y in ys_tgt]
        else:
            ys_tgt = [to_cuda(self, torch.from_numpy(y)) for y in ys_tgt]

        if self.st == 0 and self.mt > 0:
            filtered_index_mt = filter(lambda i: len(tids_src[i]) > 0, range(len(tids_src)))
            sorted_index_mt = sorted(filtered_index_mt, key=lambda i: -len(tids_src[i]))
            if len(sorted_index_mt) != len(tids_src):
                logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                    len(tids_src), len(sorted_index_mt)))

            ys_src_mt = [np.fromiter(map(int, tids_src[i]), dtype=np.int64)
                         for i in sorted_index_mt]
            if torch_is_old:
                ys_src_mt = [to_cuda(self, Variable(torch.from_numpy(y), volatile=True))
                             for y in ys_src_mt]
            else:
                ys_src_mt = [to_cuda(self, torch.from_numpy(y)) for y in ys_src_mt]

            olens = np.fromiter((len(xx) + 2 for xx in tids_src), dtype=np.int64)
            olens = [olens[i] for i in sorted_index_mt]

            # append <SOS> and <EOS>
            eos = Variable(ys_src_mt[0].data.new([self.eos]))
            sos = Variable(ys_src_mt[0].data.new([self.sos]))
            ys_src_mt = [torch.cat([sos, y, eos], dim=0) for y in ys_src_mt]
            ys_src_mt_pad = pad_list(ys_src_mt, self.eos)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        if torch_is_old:
            hs = [to_cuda(self, Variable(torch.from_numpy(xx), volatile=True)) for xx in xs]
        else:
            hs = [to_cuda(self, torch.from_numpy(xx)) for xx in xs]

        # encoder
        if self.st > 0:
            xpad = pad_list(hs, 0.0)
            hpad, hlens = self.enc(xpad, ilens)
        else:
            eys_mt = self.embed_mt(ys_src_mt_pad)
            hpad_mt, hlens_mt = self.enc_mt(eys_mt, olens)

        # decoder
        if self.st > 0:
            att_ws = self.dec_st.calculate_all_attentions(hpad, hlens, ys_tgt)
        else:
            att_ws = self.dec_mt.calculate_all_attentions(hpad_mt, hlens_mt, ys_tgt)

        if not torch_is_old:
            torch.set_grad_enabled(True)

        return att_ws


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class MTDecoder(torch.nn.Module):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0,
                 input_feeding=False):
        super(MTDecoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.ignore_id = -1
        if input_feeding:
            self.mlp_input_feeding = torch.nn.Linear(dunits + eprojs, eprojs)
            self.output = torch.nn.Linear(eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits + eprojs, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.input_feeding = input_feeding

    def zero_state(self, hpad):
        return Variable(hpad.data.new(hpad.size(0), self.dunits).zero_())

    def forward(self, hpad, hlen, ys):
        '''Decoder forward

        :param hs:
        :param ys:
        :return:
        '''
        hpad = mask_by_length(hpad, hlen, 0)
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = Variable(ys[0].data.new([self.eos]))
        sos = Variable(ys[0].data.new([self.sos]))
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = pad_list(ys_in, self.eos)
        pad_ys_out = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = pad_ys_out.size(0)
        olength = pad_ys_out.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlen))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h
        z_if = self.zero_state(hpad)

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.embed(torch.max(self.output(z_if), dim=1)[1]).detach()
                if self.input_feeding:
                    ey = torch.cat((z_out, z_if), dim=1)  # utt x (zdim + hdim)
                else:
                    ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                if self.input_feeding:
                    ey = torch.cat((eys[:, i, :], z_if), dim=1)  # utt x (zdim + hdim)
                else:
                    ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_if = torch.cat((z_list[-1], att_c), dim=1)
            if self.input_feeding:
                z_if = F.tanh(self.mlp_input_feeding(z_if))
            z_all.append(z_if)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(y_all, pad_ys_out.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, pad_ys_out, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.data).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = y_all.view(batch, olength, -1)
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data.cpu().numpy()),
                                            y_true.data.cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat_[y_true_ != self.ignore_id], axis=1)
                idx_true = y_true_[y_true_ != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, Variable(torch.from_numpy(self.labeldist)))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param Variable h:
        :param Namespace recog_args:
        :param char_list:
        :return:
        '''
        logging.info('input lengths: ' + str(h.size(0)))
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att.reset()  # reset pre-computation of h
        z_if = self.zero_state(h.unsqueeze(0))

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        # ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.sos
        if torch_is_old:
            vy = Variable(h.data.new(1).zero_().long(), volatile=True)
        else:
            vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        # if lpz is not None:
        #     ctc_prefix_score = CTCPrefixScore(lpz.numpy(), 0, self.eos, np)
        #     hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
        #     hyp['ctc_score_prev'] = 0.0
        #     if ctc_weight != 1.0:
        #         # pre-pruning based on attention scores
        #         ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
        #     else:
        #         ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)           # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.att(h.unsqueeze(0), [h.size(0)], hyp['z_prev'][0], hyp['a_prev'])
                if self.input_feeding:
                    ey = torch.cat((ey, z_if), dim=1)  # utt(1) x (zdim + hdim)
                else:
                    ey = torch.cat((ey, att_c), dim=1)   # utt(1) x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # get nbest local scores and their ids
                z_if = torch.cat((z_list[-1], att_c), dim=1)
                if self.input_feeding:
                    z_if = F.tanh(self.mlp_input_feeding(z_if))
                local_att_scores = F.log_softmax(self.output(z_if), dim=1).data
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                # if lpz is not None:
                #     local_best_scores, local_best_ids = torch.topk(
                #         local_att_scores, ctc_beam, dim=1)
                #     ctc_scores, ctc_states = ctc_prefix_score(
                #         hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                #     local_scores = \
                #         (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                #         + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                #     if rnnlm:
                #         local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                #     local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                #     local_best_ids = local_best_ids[:, joint_best_ids[0]]
                # else:
                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    # if lpz is not None:
                    #     new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                    #     new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]
        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def calculate_all_attentions(self, hpad, hlen, ys):
        '''Calculate all of attentions

        :return: numpy array format attentions
        '''
        hlen = list(map(int, hlen))
        hpad = mask_by_length(hpad, hlen, 0)
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = Variable(ys[0].data.new([self.eos]))
        sos = Variable(ys[0].data.new([self.sos]))
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = pad_list(ys_in, self.eos)
        pad_ys_out = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = pad_ys_out.size(1)

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        if isinstance(self.att, AttLoc2D):
            # att_ws => list of previous concate attentions
            att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, (AttCov, AttCovLoc)):
            # att_ws => list of list of previous attentions
            att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, AttLocRec):
            # att_ws => list of tuple of attention and hidden states
            att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc)):
            # att_ws => list of list of each head attetion
            n_heads = len(att_ws[0])
            att_ws_sorted_by_head = []
            for h in six.moves.range(n_heads):
                att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
                att_ws_sorted_by_head += [att_ws_head]
            att_ws = torch.stack(att_ws_sorted_by_head, dim=1).data.cpu().numpy()
        else:
            # att_ws => list of attetions
            att_ws = torch.stack(att_ws, dim=1).data.cpu().numpy()
        return att_ws
