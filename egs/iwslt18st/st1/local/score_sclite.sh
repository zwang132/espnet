#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
lc=false

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn.py ${dir}/data.json ${dic} ${dir}/ref.trn ${dir}/hyp.trn

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
fi
if ${lc}; then
  awk '{ print tolower($0) }' < ${dir}/hyp.trn > ${dir}/hyp.trn.tmp
  awk '{ print tolower($0) }' < ${dir}/ref.trn > ${dir}/ref.trn.tmp
  mv ${dir}/hyp.trn.tmp ${dir}/hyp.trn; rm ${dir}/hyp.trn.tmp
  mv ${dir}/ref.trn.tmp ${dir}/ref.trn; rm ${dir}/ref.trn.tmp
fi

# detokenize
detokenizer.perl -l en < ${dir}/ref.trn > ${dir}/ref.trn.detok
detokenizer.perl -l en < ${dir}/hyp.trn > ${dir}/hyp.trn.detok

sclite -r ${dir}/ref.trn.detok trn -h ${dir}/hyp.trn.detok trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    if [ ! -z $bpe ]; then
      spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
      spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    else
      sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
      sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    fi

    # detokenize
    detokenizer.perl -l en < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
    detokenizer.perl -l en < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok

    sclite -r ${dir}/ref.wrd.trn.detok trn -h ${dir}/hyp.wrd.trn.detok trn -i rm -o all stdout > ${dir}/result.wrd.txt

    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
