#!/bin/bash

# Copyright 2018  Johns Hopkins University (Author: Zhiqi Wang)
# Apache 2.0.

dir=$1
decode_dir=$2
local=`pwd`/local

# Extract error sentences
$local/extract_error.py $decode_dir | sed 's/([0-9]/\|/g' | cut -f 1 -d'|' | sed 's/ *$//' > $dir/error.txt
