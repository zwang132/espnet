#!/usr/bin/env python

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import logging
import os

import numpy as np

# chainer related
from chainer.training import extension

# io related
import kaldi_io_py

# matplotlib related
import matplotlib
matplotlib.use('Agg')


def make_batchset(data, batch_size, max_length_in, max_length_out, num_batches=0):
    # sort it by input lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['input'][0]['shape'][0]), reverse=True)
    logging.info('# utts: ' + str(len(sorted_data)))
    # change batchsize depending on the input and output length
    minibatch = []
    start = 0
    while True:
        ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
        olen = int(sorted_data[start][1]['output'][0]['shape'][0])
        factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(1, .) avoids batchsize = 0
        b = max(1, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + b)
        minibatch.append(sorted_data[start:end])
        if end == len(sorted_data):
            break
        start = end
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch


def load_inputs_and_targets(batch, sort_in_outputs=False, asr=0, mt=0):
    """Function to load inputs and targets from list of dicts

    :param list batch: list of dict which is subset of loaded data.json
    :param bool sort_in_outputs: whether to sort in output lengths
    :param bool use_speaker_embedding: whether to load speaker embedding vector
    :return: list of input feature sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of target token id sequences [(T_1), (T_2), ..., (T_B)]
    :rtype: list of int ndarray
    :return: list of speaker embedding vectors (only if use_speaker_embedding = True)
    :rtype: list of float adarray
    """

    # load acoustic features and target sequence of token ids
    xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]
    ys = [b[1]['output'][0]['tokenid'].split() for b in batch]
    if asr > 0 or mt > 0:
        ys_src = [b[1]['output'][1]['tokenid'].split() for b in batch]

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(xs)))
    if sort_in_outputs:
        # sort in output lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(ys[i]))
    else:
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
            len(xs), len(nonzero_sorted_idx)))
    if mt > 0:
        nonzero_idx_mt = filter(lambda i: len(ys[i]) > 0, range(len(ys_src)))
        # sort in input lengths
        nonzero_sorted_idx_mt = sorted(nonzero_idx_mt, key=lambda i: -len(ys_src[i]))
        if len(nonzero_sorted_idx_mt) != len(ys_src):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(ys_src), len(nonzero_sorted_idx_mt)))

    # remove zero-length samples
    xs_mt = [np.fromiter(map(int, ys_src[i]), dtype=np.int64)
             for i in nonzero_sorted_idx_mt] if mt > 0 else None
    ys_mt = [np.fromiter(map(int, ys[i]), dtype=np.int64)
             for i in nonzero_sorted_idx_mt] if mt > 0 else None
    # NOTE: this must be done before the next ys loop
    xs = [xs[i] for i in nonzero_sorted_idx]
    ys = [np.fromiter(map(int, ys[i]), dtype=np.int64)
          for i in nonzero_sorted_idx]
    ys_asr = [np.fromiter(map(int, ys_src[i]), dtype=np.int64)
              for i in nonzero_sorted_idx] if asr > 0 else None

    others = {
        "ys_asr": ys_asr,
        "xs_mt": xs_mt,
        "ys_mt": ys_mt,
    }

    return xs, ys, others


class PlotAttentionReport(extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param function converter: function to convert data
    :param int device: device id
    :param bool reverse: If True, input and output length are reversed
    """

    def __init__(self, att_vis_fn, data, outdir, converter, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        att_ws_ = self.att_vis_fn(*batch)
        for task, att_ws in att_ws_.items():
            if att_ws is None:
                continue
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.ep.{.updater.epoch}_%s.png" % (
                    self.outdir, self.data[idx][0], task)
                if self.reverse:
                    if task == 'mt':
                        dec_len = int(self.data[idx][1]['output'][1]['shape'][0]) + 2
                        enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
                    else:
                        dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
                        if task == 'st':
                            enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
                        if task == 'asr':
                            enc_len = int(self.data[idx][1]['output'][1]['shape'][0])
                else:
                    if task == 'mt':
                        dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
                        enc_len = int(self.data[idx][1]['output'][1]['shape'][0]) + 2
                    else:
                        if task == 'st':
                            dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
                        if task == 'asr':
                            dec_len = int(self.data[idx][1]['output'][1]['shape'][0])
                        enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
                if len(att_w.shape) == 3:
                    att_w = att_w[:, :dec_len, :enc_len]
                else:
                    att_w = att_w[:dec_len, :enc_len]
                self._plot_and_save_attention(att_w, filename.format(trainer))

    def _plot_and_save_attention(self, att_w, filename):
        # dynamically import matplotlib due to not found error
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
