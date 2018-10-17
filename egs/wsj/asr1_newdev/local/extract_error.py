#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (Zhiqi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys

if __name__ == '__main__':
    hypdir = sys.argv[1] + '/hyp.wrd.trn'
    refdir = sys.argv[1] + '/ref.wrd.trn'
    with open(hypdir, 'r') as f:
        hyp = f.readlines()
    with open(refdir, 'r') as h:
        ref = h.readlines()
    for i in range(len(hyp)):
        if hyp[i].strip() != ref[i].strip():
            sys.stdout.write(hyp[i])
