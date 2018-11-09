#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network archtecture
# encoder related
etype=blstm     # encoder architecture type
elayers=2
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=2
dunits=1024
# attention related
atype=add
adim=1024
aconv_chans=10
aconv_filts=100

# regularization related
samp_prob=0
dropout=0

# input feeding option
input_feeding=true

# cold_fusion
cold_fusion=

# minibatch related
batchsize=64
maxlen_in=100  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=100 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20

# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=5
penalty=1.0
maxlenratio=1.0
minlenratio=0.0
recog_model=model.ppl.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best' or 'model.ppl.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/b08/inaguma/IWSLT


# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# data directories
iwsltdir=../../iwslt18st
wmtdir=/export/a06/kduh/iwslt18/

train_set=train_plus_wmt18.de
train_dev=dev.de
recog_set="dev2010.de tst2010.de tst2013.de tst2014.de tst2015.de"


dict=data/lang_1spm/train_${bpemode}${nbpe}_units.txt
nlsyms=data/lang_1spm/non_lang_syms.txt
bpemodel=data/lang_1spm/train_${bpemode}${nbpe}

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep_train.sh ${datadir}
    for part in dev2010 tst2010 tst2013 tst2014 tst2015 tst2018; do
        local/data_prep_eval.sh ${datadir} ${part}
    done

    for lang in en de; do
      mkdir -p data/wmt18_org.${lang}

      # normalize punctuation & tokenize
      cat ${wmtdir}/wmt-mlsubset.${lang} | normalize-punctuation.perl -l ${lang} data/wmt18_org.${lang}/text | \
        tokenizer.perl -a -l ${lang} \
        | awk '{ printf("utt%07d %s\n", NR, $0) }' > data/wmt18_org.${lang}/text || echo ''

      # remove noisy samples (hindi etc.)
      paste -d " " <(awk '{print $1}' data/wmt18_org.${lang}/text) <(cut -f 2- -d " " data/wmt18_org.${lang}/text \
        | spm_encode --model=${bpemodel}.model --output_format=piece) \
        > data/wmt18_org.${lang}/text.token
      cat data/wmt18_org.${lang}/text.token | utils/sym2int.pl --map-oov '<unk>' -f 2- ${dict} \
        > data/wmt18_org.${lang}/text.tokenid
      cat data/wmt18_org.${lang}/text.tokenid | awk '{
          num_oov=0;
          for (i=2; i<NF; i++) {if ($i == 1) num_oov+=1};
          oov_rate=num_oov*100/(NF - 1);
          if (oov_rate  <= 10) printf("%s\n", $0);
      }' > data/wmt18_org.${lang}/text.tokenid_clean
      # NOTE: remove sentences whose OOV rate are more then 10%

      cat data/wmt18_org.${lang}/text.tokenid_clean | awk '{ printf("%s spk1\n", $1); }' \
        > data/wmt18_org.${lang}/utt2spk

      # remove utt having more than 100 tokens
      cat data/wmt18_org.${lang}/text.tokenid_clean | awk '{ if (NF <= 100) print $1; }' | cut -f -1 -d " " \
        > data/wmt18_org.${lang}/reclist.${lang}
    done

    # Match the number of utterances between En and De
    # extract commocn lines
    comm -12 data/wmt18_org.en/reclist.en data/wmt18_org.de/reclist.de > data/wmt18_org.de/reclist

    reduce_data_dir.sh data/wmt18_org.en data/wmt18_org.de/reclist data/wmt18.en
    reduce_data_dir.sh data/wmt18_org.de data/wmt18_org.de/reclist data/wmt18.de
    rm -rf data/wmt18_org.en data/wmt18_org.de

    # combine the training set
    for lang in en de; do
      mkdir -p data/train.${lang}.tmp
      cp data/train.${lang}/text data/train.${lang}.tmp
      cp data/train.${lang}/spk2utt data/train.${lang}.tmp
      cp data/train.${lang}/utt2spk data/train.${lang}.tmp
      utils/combine_data.sh --skip_fix true data/train_plus_wmt18.${lang} data/wmt18.${lang} data/train.${lang}.tmp
      rm -rf data/train.${lang}.tmp
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}


echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    if [ ! -f ${dict} ]; then
      echo "run local/run_spm.sh first"
      exit 1
    fi

    if [ ! -f ${dict} ]; then
      echo "run local/run_spm.sh first"
      exit 1
    fi

    # make json labels
    data2json.sh --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}_mt.json
    data2json.sh --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}_mt.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --utt2spk false --text data/${rtask}/text_noseg --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}_mt.json
    done

    # Update json (Add En)
    for name in ${train_set} ${train_dev}; do
        feat_dir=${dumpdir}/${name}/delta${do_delta}
        data_dir=data/`echo ${name} | cut -f -1 -d "."`.en
        local/update_json.sh --bpecode ${bpemodel}.model ${feat_dir}/data_${bpemode}${nbpe}_mt.json ${data_dir} ${dict}
    done
    for rtask in ${recog_set}; do
        feat_dir=${dumpdir}/${rtask}/delta${do_delta}
        data_dir=data/`echo ${rtask} | cut -f -1 -d "."`.en
        local/update_json.sh --text ${data_dir}/text_noseg --bpecode ${bpemodel}.model ${feat_dir}/data_${bpemode}${nbpe}_mt.json ${data_dir} ${dict}
    done
fi


# You can skip this and remove --rnnlm option in the recognition (stage 3)
lmexpdir=exp/${train_set}_rnnlm_${backend}_2layer_bs256_${bpemode}${nbpe}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_${train_set}_${bpemode}${nbpe}
    mkdir -p ${lmdatadir}
    cut -f 2- -d " " data/${train_set}/text | spm_encode --model=${bpemodel}.model --output_format=piece | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train.txt
    cut -f 2- -d " " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --epoch 60 \
        --batchsize 256 \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_${bpemode}${nbpe}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
    if [ ! -z ${input_feeding} ]; then
        expdir=${expdir}_inputfeed
    fi
    if [ ${dropout} != 0 ]; then
        expdir=${expdir}_drop${dropout}
    fi
    if [ -n "${cold_fusion}" ]; then
      expdir=${expdir}_cf${cold_fusion}
      if [ `echo ${lmexpdir} | grep 'wmt18'` ] ; then
          expdir=${expdir}_wmt18
      fi
    fi
    expdir=${expdir}_mt
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        mt_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}_mt.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}_mt.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --dropout-rate ${dropout} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --epochs ${epochs} \
        --input-feeding ${input_feeding} \
        --rnnlm-cf ${lmexpdir}/rnnlm.model.best \
        --cold-fusion ${cold_fusion}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_rnnlm${lm_weight}
        mkdir -p ${expdir}/${decode_dir}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}_mt.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}_mt.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --lm-weight ${lm_weight} \
            &
            # --rnnlm ${lmexpdir}/rnnlm.model.best \
        wait

        set=`echo ${rtask} | cut -f -1 -d "."`
        local/score_bleu_mt.sh --nlsyms ${nlsyms} --bpe ${nbpe} --bpemodel ${bpemodel}.model ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
