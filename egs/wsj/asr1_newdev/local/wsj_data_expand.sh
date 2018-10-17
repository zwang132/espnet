#!/bin/bash

# Copyright 2018  Johns Hopkins University (Author: Zhiqi Wang)
# Apache 2.0.

lmdatadir=$1
local=`pwd`/local

# Extract error sentences
$local/shuffle_word.py ${lmdatadir}_tmp/train.txt ${lmdatadir}/train_shuffle.txt
