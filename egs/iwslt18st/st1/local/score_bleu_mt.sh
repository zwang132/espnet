#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
word=false
bpe=""
bpemodel=""
filter=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <decode-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

# sort hypotheses
concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_mt.py ${dir}/data.json ${dic} ${dir}/ref.en.trn ${dir}/ref.de.trn ${dir}/hyp.de.trn

spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.de.trn | sed -e "s/▁/ /g" > ${dir}/ref.de.wrd.trn
spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.de.trn | sed -e "s/▁/ /g" > ${dir}/hyp.de.wrd.trn

# detokenize
detokenizer.perl -u -l de < ${dir}/ref.de.wrd.trn > ${dir}/ref.de.wrd.trn.detok
detokenizer.perl -u -l de < ${dir}/hyp.de.wrd.trn > ${dir}/hyp.de.wrd.trn.detok
# NOTE: uppercase the first character (-u)

### case-insensitive
multi-bleu-detok.perl -lc ${dir}/ref.de.wrd.trn.detok < ${dir}/hyp.no-case.wrd.trn.detok > ${dir}/result.no-case.wrd.txt
echo "write a case-insensitive word-level BLUE result in ${dir}/result.no-case.wrd.txt"
cat ${dir}/result.no-case.wrd.txt

### case-sensitive
multi-bleu-detok.perl ${dir}/ref.de.wrd.trn.detok < ${dir}/hyp.de.wrd.trn.detok > ${dir}/result.wrd.txt
echo "write a case-sensitve word-level BLUE result in ${dir}/result.wrd.txt"
cat ${dir}/result.wrd.txt


# TODO(hirofumi): add TER & METEOR metrics here
