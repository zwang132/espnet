#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <decode-dir> <data-dir> <set>";
    exit 1;
fi

dir=$1
set=$3
src=$2/$set/IWSLT.$set
system=mt

mkdir -p ${dir}/${set}

cp ${dir}/translation_${set} ${dir}/${set}/hyp.wrd.trn

# generate reference
xml_en=$src/IWSLT.TED.$set.en-de.en.xml
xml_de=$src/IWSLT.TED.$set.en-de.de.xml

grep "<seg id" $xml_de | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/${set}/ref.wrd.trn


# detokenize
detokenizer.perl -u -l de < ${dir}/${set}/hyp.wrd.trn > ${dir}/${set}/hyp.wrd.trn.detok
# NOTE: uppercase the first character (-u)

### case-sensitive
# segment hypotheses with RWTH tool
# segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/${set}/hyp.wrd.trn.detok $system de ${dir}/${set}/hyp.wrd.trn.detok.sgm.xml "" 1
segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/${set}/hyp.wrd.trn.detok $system de ${dir}/${set}/hyp.wrd.trn.detok.sgm.xml "" 1
sed -e "/<[^>]*>/d" ${dir}/${set}/hyp.wrd.trn.detok.sgm.xml > ${dir}/${set}/hyp.wrd.trn.detok.sgm


multi-bleu-detok.perl ${dir}/${set}/ref.wrd.trn < ${dir}/${set}/hyp.wrd.trn.detok.sgm > ${dir}/${set}/result.wrd.txt
echo "write a case-sensitve word-level BLUE result in ${dir}/${set}/result.wrd.txt"
cat ${dir}/${set}/result.wrd.txt
