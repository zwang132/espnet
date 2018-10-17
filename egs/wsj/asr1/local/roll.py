#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (Zhiqi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        hyp = f.readlines()
    with open(sys.argv[2], 'r') as h:
        ref = h.readlines()
    res = open('result.trn', 'w')
    for i in range(len(hyp)):
        if hyp[i].strip() != ref[i].strip():
            res.write(hyp[i])
    res.close()
