#!/usr/bin/env python

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import logging
import os
import sys

# chainer related
import chainer

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

# torch related
import torch

# espnet related
from asr_utils import adadelta_eps_decay
from asr_utils import add_results_to_json
from asr_utils import CompareValueTrigger
from asr_utils import get_model_conf
from st_utils import load_inputs_and_targets
from asr_utils import load_labeldict
from asr_utils import make_batchset
from st_utils import make_batchset
from st_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from asr_utils import torch_load
from asr_utils import torch_resume
from asr_utils import torch_save
from asr_utils import torch_snapshot
from e2e_st_th import E2E
from e2e_st_th import Loss
from e2e_asr_th import pad_list

# from e2e_asr_th import E2E as E2E_asr
# from e2e_asr_th import Loss as Loss_asr
from e2e_st_th import E2E as E2E_asr
from e2e_st_th import Loss as Loss_asr
from e2e_mt_th import E2E as E2E_mt
from e2e_mt_th import Loss as Loss_mt

# for kaldi io
import kaldi_io_py

from asr_pytorch import CustomEvaluator
from asr_pytorch import CustomUpdater


# rnnlm
import extlm_pytorch
import lm_pytorch

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

REPORT_INTERVAL = 100


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1, odim=0, asr=0, mt=0):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        self.asr = asr
        self.mt = mt

    def transform(self, item):
        return load_inputs_and_targets(item, asr=self.asr, mt=self.mt)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys, others = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long()for y in ys], self.ignore_id).to(device)
        ys_asr_pad = pad_list([torch.from_numpy(y).long()
                               for y in others['ys_asr']], self.ignore_id).to(device) if self.asr > 0 else None
        if self.mt > 0:
            # get batch of lengths of input sequences
            ilens_mt = np.array([len(x) + 2 for x in others['xs_mt']])
            # NOTE: +2 means <SOS> and <EOS> tokens

            ilens_mt = torch.from_numpy(ilens_mt).to(device)

            # prepare input and output word sequences with sos/eos IDs
            xs_mt_pad = [torch.from_numpy(x).long()for x in others['xs_mt']]
            eos = xs_mt_pad[0].new([self.eos])
            sos = xs_mt_pad[0].new([self.sos])
            xs_mt_pad = [torch.cat([sos, x], dim=0) for x in xs_mt_pad]
            xs_mt_pad = [torch.cat([x, eos], dim=0) for x in xs_mt_pad]

            xs_mt_pad = pad_list(xs_mt_pad, self.eos).to(device)
            ys_mt_pad = pad_list([torch.from_numpy(y).long()
                                  for y in others['ys_mt']], self.ignore_id).to(device)
        else:
            xs_mt_pad, ilens_mt, ys_mt_pad = None, None, None

        others_pad = {
            "ys_asr": ys_asr_pad,
            "xs_mt": xs_mt_pad,
            "ilens_mt": ilens_mt,
            "ys_mt": ys_mt_pad,
        }
        return xs_pad, ilens, ys_pad, others_pad


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # Initialize encoder with pre-trained ASR encoder
    if args.asr_model:
        # read training config
        idim_asr, odim_asr, train_args_asr = get_model_conf(args.asr_model)
        e2e_asr = E2E_asr(idim_asr, odim_asr, train_args_asr)
        asr_model = Loss_asr(e2e_asr, train_args_asr.mtlalpha)
        torch_load(args.asr_model, asr_model)
    else:
        asr_model = None

    # Initialize decoder with pre-trained MT decoder
    if args.mt_model:
        idim_mt, odim_mt, train_args_mt = get_model_conf(args.mt_model)
        e2e_mt = E2E_mt(idim_mt, odim_mt, train_args_mt)
        mt_model = Loss_mt(e2e_mt)
        torch_load(args.mt_model, mt_model)
    else:
        mt_model = None

    # read rnnlm for cold fusion
    if args.cold_fusion:
        assert args.rnnlm_cf is not None
        rnnlm_cf_args = get_model_conf(args.rnnlm_cf)
        rnnlm_cf = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(args.char_list), rnnlm_cf_args.unit))
        torch_load(args.rnnlm_cf, rnnlm_cf)
        # rnnlm_cf.eval()
    else:
        rnnlm_cf = None

    # specify model architecture
    e2e = E2E(idim, odim, args, asr_model, mt_model, rnnlm_cf)
    model = Loss(e2e, args.asr, args.mt, args.mtlalpha)

    if args.asr_model:
        del asr_model

    if args.mt_model:
        del mt_model

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    # NOTE: exclude fixed parameters (eg. cold fusion)
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            parameters, rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            parameters)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(e2e.subsample[0], odim, args.asr, args.mt)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    #train_iter = chainer.iterators.MultiprocessIterator(
    #    TransformDataset(train, converter.transform),
    #    batch_size=1, n_processes=1, n_prefetch=8)
    # maxtasksperchild=20
    #valid_iter = chainer.iterators.SerialIterator(
    #    TransformDataset(valid, converter.transform),
    #    batch_size=1, repeat=False, shuffle=False)
    train_iter = chainer.iterators.SerialIterator(
        TransformDataset(train, converter.transform),
        batch_size=1)
    valid_iter = chainer.iterators.SerialIterator(
        TransformDataset(valid, converter.transform),
        batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, converter, device, args.ngpu)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        # sort it by output lengths
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['output'][0]['shape'][0]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.predictor.calculate_all_attentions
        else:
            att_vis_fn = model.predictor.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, device=device), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_st', 'validation/main/loss_st',
                                          'main/loss_asr', 'validation/main/loss_asr',
                                          'main/loss_mt', 'validation/main/loss_mt'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc',
                                          'main/acc_asr', 'validation/main/acc_asr',
                                          'main/acc_mt', 'validation/main/acc_mt'],
                                         'epoch', file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['main/ppl', 'validation/main/ppl',
                                          'main/ppl_mt', 'validation/main/ppl_mt'],
                                         'epoch', file_name='ppl.png'))

    # Save best models
    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))
        trainer.extend(extensions.snapshot_object(model, 'model.ppl.best', savefun=torch_save),
                       trigger=training.triggers.MinValueTrigger('validation/main/ppl'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
        elif args.criterion == 'ppl':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.ppl.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/ppl',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/ppl',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_st', 'main/loss_asr', 'main/loss_mt',
                   'validation/main/loss', 'validation/main/loss_st', 'validation/main/loss_asr', 'validation/main/loss_mt',
                   'main/acc', 'main/acc_asr', 'main/acc_mt',
                   'validation/main/acc', 'validation/main/acc_asr', 'validation/main/acc_mt',
                   'main/ppl', 'validation/main/ppl',
                   'main/ppl_mt', 'validation/main/ppl_mt',
                   'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(REPORT_INTERVAL, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # read rnnlm for cold fusion
    if train_args.cold_fusion:
        assert train_args.rnnlm_cf is not None
        rnnlm_cf_args = get_model_conf(train_args.rnnlm_cf)
        rnnlm_cf = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), rnnlm_cf_args.unit))
        torch_load(train_args.rnnlm_cf, rnnlm_cf)
        rnnlm_cf.eval()
    else:
        rnnlm_cf = None

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args, rnnlm_cf=rnnlm_cf)
    model = Loss(e2e, train_args.asr, train_args.mt, train_args.mtlalpha)
    torch_load(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        if not args.word_dict:
            logging.error('word dictionary file is not specified for the word RNNLM.')
            sys.exit(1)

        rnnlm_args = get_model_conf(args.word_rnnlm, args.rnnlm_conf)
        word_dict = load_labeldict(args.word_dict)
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(len(word_dict), rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
            nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
