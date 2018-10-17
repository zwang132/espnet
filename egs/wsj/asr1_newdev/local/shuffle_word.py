#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (Zhiqi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import random

def Shuffle(s, filename):
    f = open(filename, 'w')
    for line in s:
        w = line.strip().split()
        new_line = []
        for word in w:
            if word.isalpha():
                l = list(word)
                random.shuffle(l)
                new_line.append(''.join(l))
            else:
                new_line.append(word)
        f.write(' '.join(new_line) + '\n')
    f.close()

if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    sentences = f.readlines()
    f.close()
    Shuffle(sentences, sys.argv[2])
