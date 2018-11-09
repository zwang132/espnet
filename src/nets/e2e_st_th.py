#!/usr/bin/env python

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import sys

import chainer
import numpy as np
import six
import torch

from chainer import reporter

from e2e_asr_common import label_smoothing_dist

from e2e_asr_th import AttAdd
from e2e_asr_th import AttCov
from e2e_asr_th import AttCovLoc
from e2e_asr_th import AttDot
from e2e_asr_th import AttLoc
from e2e_asr_th import AttLoc2D
from e2e_asr_th import AttLocRec
from e2e_asr_th import AttMultiHeadAdd
from e2e_asr_th import AttMultiHeadDot
from e2e_asr_th import AttMultiHeadLoc
from e2e_asr_th import AttMultiHeadMultiResLoc
from e2e_asr_th import CTC
from e2e_asr_th import Decoder as ASRDecoder
from e2e_mt_th import Decoder as MTDecoder
from e2e_asr_th import Encoder
from e2e_asr_th import NoAtt

from e2e_asr_th import to_cuda


torch_is_old = torch.__version__.startswith("0.3.")

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


class Reporter(chainer.Chain):
    def report(self, loss_st, loss_asr, loss_mt,
               acc_st, acc_asr, acc_mt, ppl_st, ppl_mt, mtl_loss):
        reporter.report({'loss_st': loss_st}, self)
        reporter.report({'loss_asr': loss_asr}, self)
        reporter.report({'loss_mt': loss_mt}, self)
        reporter.report({'acc': acc_st}, self)
        reporter.report({'acc_asr': acc_asr}, self)
        reporter.report({'acc_mt': acc_mt}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)
        reporter.report({'ppl': ppl_st}, self)
        reporter.report({'ppl_mt': ppl_mt}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    """Multi-task learning loss module

    :param torch.nn.Module predictor: E2E model instance
    :param float mtlalpha: mtl coefficient value (0.0 ~ 1.0)
    """

    def __init__(self, predictor, asr, mt, mtlalpha):
        super(Loss, self).__init__()
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha shoule be [0.0, 1.0]"
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

        # for MTL
        self.st = 1 - asr - mt
        self.asr = asr
        self.mt = mt

    def forward(self, xs_pad, ilens, ys_pad, others_pad):
        '''Multi-task learning loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        loss_st, loss_asr, loss_mt, acc_st, acc_asr, acc_mt, ppl_st, ppl_mt = self.predictor(
            xs_pad, ilens, ys_pad, others_pad)

        self.loss = self.st * loss_st
        loss_st_data = float(loss_st)

        if self.asr > 0:
            # MTL with ASR
            self.loss += self.asr * loss_asr
            loss_asr_data = float(loss_asr)
        elif self.mtlalpha > 0:
            # pure ASR mode
            self.loss = (1 - self.mtlalpha) * loss_st + self.mtlalpha * loss_asr
            loss_asr_data = float(loss_asr)  # means ctc loss
        else:
            loss_asr_data = None

        if self.mt > 0:
            self.loss += self.mt * loss_mt
            loss_mt_data = float(loss_mt)
        else:
            loss_mt_data = None

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_st_data, loss_asr_data, loss_mt_data,
                                 acc_st, acc_asr, acc_mt, ppl_st, ppl_mt, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim, args,
                 asr_model=None, mt_model=None, rnnlm_cf=None):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha

        # for MTL
        self.st = 1 - args.asr - args.mt
        self.asr = args.asr
        self.asr_type = args.asr_type
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
        self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                           self.subsample, args.dropout_rate)
        # attention
        if args.atype == 'noatt':
            self.att = NoAtt()
        elif args.atype == 'dot':
            self.att = AttDot(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'add':
            self.att = AttAdd(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'location':
            self.att = AttLoc(args.eprojs, args.dunits,
                              args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location2d':
            self.att = AttLoc2D(args.eprojs, args.dunits,
                                args.adim, args.awin, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location_recurrent':
            self.att = AttLocRec(args.eprojs, args.dunits,
                                 args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'coverage':
            self.att = AttCov(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'coverage_location':
            self.att = AttCovLoc(args.eprojs, args.dunits, args.adim,
                                 args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_dot':
            self.att = AttMultiHeadDot(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_add':
            self.att = AttMultiHeadAdd(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_loc':
            self.att = AttMultiHeadLoc(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim,
                                       args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_multi_res_loc':
            self.att = AttMultiHeadMultiResLoc(args.eprojs, args.dunits,
                                               args.aheads, args.adim, args.adim,
                                               args.aconv_chans, args.aconv_filts)
        else:
            logging.error(
                "Error: need to specify an appropriate attention archtecture")
            sys.exit()
        # decoder
        self.dec = MTDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                             self.sos, self.eos, self.att, self.verbose, self.char_list,
                             labeldist, args.lsm_weight, args.sampling_probability,
                             args.input_feeding, rnnlm_cf, args.cold_fusion, args.replace_sos)

        # for ASR
        if args.asr > 0:
            # MTL with ASR
            if args.asr_type == 'ctc':
                self.ctc_asr = CTC(odim, args.eprojs, args.dropout_rate)
            elif args.asr_type == 'att':
                self.att_asr = AttLoc(args.eprojs, args.dunits,
                                      args.adim, args.aconv_chans, args.aconv_filts)
                self.dec_asr = ASRDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                                          self.sos, self.eos, self.att_asr, self.verbose, self.char_list,
                                          labeldist, args.lsm_weight, args.sampling_probability)
                # self.dec_asr = MTDecoder(args.eprojs, odim, args.dlayers, args.dunits,
                #                          self.sos, self.eos, self.att, self.verbose, self.char_list,
                #                          labeldist, args.lsm_weight, args.sampling_probability,
                #                          args.input_feeding, rnnlm_cf, args.cold_fusion, args.replace_sos)
        elif self.mtlalpha > 0:
            # pure ASR mode
            self.ctc_asr = CTC(odim, args.eprojs, args.dropout_rate)

        # for MT
        if args.mt > 0:
            self.enc_mt = Encoder(args.etype, args.dunits, 2, args.eunits, args.eprojs,
                                  [1, 1, 1], args.dropout_rate)
            self.embed_in = self.dec.embed
            # NOTE: share embedding between inputs and outputs

        # weight initialization
        self.init_like_chainer()

        # transfer learning from pre-trained models
        self.init_pretrained_model(asr_model, mt_model)

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for n, p in module.named_parameters():
                # Skip pre-trained RNNLM for cold fusion
                if 'rnnlm_cf' in n:
                    continue

                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)
        # TODO(hirofumi): how about encoder??

        if self.asr > 0 and self.asr_type == 'att':
            # exceptions
            # embed weight ~ Normal(0, 1)
            self.dec_asr.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in six.moves.range(len(self.dec_asr.decoder)):
                set_forget_bias_to_one(self.dec_asr.decoder[l].bias_ih)

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
            mt_dec_param_dict = dict(mt.predictor.dec.named_parameters())
            for n, p in self.dec.named_parameters():
                p.data = mt_dec_param_dict[n].data

    def forward(self, xs_pad, ilens, ys_pad, others_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 1. encoder
        hs_pad, hlens = self.enc(xs_pad, ilens)

        # 2. ST loss
        loss_st, acc_st, ppl_st = self.dec(hs_pad, hlens, ys_pad)
        loss_st = loss_st.unsqueeze(0)

        # 3. ASR loss
        if self.asr > 0:
            # MTL with ASR
            if self.asr_type == 'ctc':
                loss_asr = self.ctc_asr(hs_pad, hlens, others_pad['ys_asr'])
                acc_asr = None
            elif self.asr_type == 'att':
                loss_asr, acc_asr = self.dec_asr(hs_pad, hlens, others_pad['ys_asr'])
                # loss_asr, acc_asr, ppl_asr = self.dec_asr(hs_pad, hlens, others_pad['ys_asr'])
                loss_asr = loss_asr.unsqueeze(0)
        else:
            # pure ASR mode
            acc_asr = None
            if self.mtlalpha > 0:
                loss_asr = self.ctc_asr(hs_pad, hlens, ys_pad)  # means ctc loss
            else:
                loss_asr = None

        # 4. MT loss
        if self.mt > 0:
            eys_mt = self.embed_in(others_pad['xs_mt'])
            hpad_mt, hlens_mt = self.enc_mt(eys_mt, others_pad['ilens_mt'])
            loss_mt, acc_mt, ppl_mt = self.dec(hpad_mt, hlens_mt, others_pad['ys_mt'])
            loss_mt = loss_mt.unsqueeze(0)
        else:
            loss_mt, acc_mt, ppl_mt = None, None, None

        return loss_st, loss_asr, loss_mt, acc_st, acc_asr, acc_mt, ppl_st, ppl_mt

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h = h.contiguous()
        h, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc_asr.log_softmax(h)[0]
        else:
            lpz = None
        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, others_pad):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            # encoder
            hs_pad, hlens = self.enc(xs_pad, ilens)

            if self.mt > 0:
                hpad_mt, hlens_mt = self.enc_mt(self.embed_in(others_pad['xs_mt']), others_pad['ilens_mt'])

            # decoder
            att_ws_st = self.dec.calculate_all_attentions(hs_pad, hlens, ys_pad)
            att_ws_asr = self.dec_asr.calculate_all_attentions(
                hs_pad, hlens, others_pad['ys_asr']) if self.asr > 0 and self.asr_type == 'att' else None
            att_ws_mt = self.dec_mt.calculate_all_attentions(
                hpad_mt, hlens_mt, others_pad['ys_mt']) if self.mt > 0 else None

        att_ws_dict = {
            "st": att_ws_st,
            "asr": att_ws_asr,
            "mt": att_ws_mt,
        }
        return att_ws_dict
