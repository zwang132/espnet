#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (Zhiqi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
from random import randint
from random import sample

def Find_Error(ref, hyp):
    error = []
    for i in range(len(ref)):
        if hyp[i].strip() != ref[i].strip():
            error.append(hyp[i])
    return error

def Get_Error_Label(error):
    label = []
    for i in range(len(error)):
        label_t = error[i].split('(')[-1][:-2].split('-')[-1]
        label.append(label_t)
    return label

def Get_Label(text):
    label = []
    for i in range(len(text)):
        label_t = text[i].split()[0]
        label.append(label_t)
    return label

def Check_Label(text, error):
    tag = []
    label_error = Get_Error_Label(error)
    label_text = Get_Label(text)
    for i in range(len(label_text)):
        for j in range(len(label_error)):
            if label_text[i] == label_error[j]:
                tag.append(i)
                break
    return tag

def Refine_Text(text, tag):
    j = 0
    refine = []
    for i in range(len(text)):
        if j < len(tag):
            if i == tag[j]:
                j += 1
            else:
                sentence = text[i].split()[1:]
                pair_number = randint(1,3)
                if pair_number < len(sentence)-1:
                    pair = sample(range(len(sentence)-1), pair_number)
                    for k in range(len(pair)):
                        sentence[pair[k]], sentence[pair[k]+1] = sentence[pair[k]+1], sentence[pair[k]]
                else:
                    if len(sentence) > 1:
                        sentence[0], sentence[1] = sentence[1], sentence[0]
                refine.append(' '.join(sentence))
        else:
            sentence = text[i].split()[1:]
            pair_number = randint(1,3)
            if pair_number < len(sentence)-1:
                pair = sample(range(len(sentence)-1), pair_number)
                for k in range(len(pair)):
                    sentence[pair[k]], sentence[pair[k]+1] = sentence[pair[k]+1], sentence[pair[k]]
            else:
                if len(sentence) > 1:
                    sentence[0], sentence[1] = sentence[1], sentence[0]
            refine.append(' '.join(sentence))
    return refine

def Concatenate_Text(text_refine, error, tag, L):
    p = 0
    q = 0
    f = open('result', 'w')
    for i in range(L):
        if q < len(tag):
            if i == tag[q]:
                f.write(error[q].split('(')[:-1][0].strip() + '\n')
                q += 1
            else:
                f.write(text_refine[p] + '\n')
                p += 1
        else:
            f.write(text_refine[p] + '\n')
            p += 1
    f.close()

def Sample_text(text, error, filename, filename2):
    f = open(filename, 'w')
    h = open(filename2, 'w')
    label_error = Get_Error_Label(error)
    label_text = Get_Label(text)
    for i in range(len(label_text)):
        for j in range(len(label_error)):
            if label_text[i] == label_error[j]:
                words = text[i].split()[1:]
                f.write(' '.join(words) + '\n')
                words = error[j].split()[:-1]
                h.write(' '.join(words) + '\n')
                break
    f.close()
    h.close()

if __name__ == '__main__':
    ref_name = sys.argv[1] + '/ref.wrd.trn'
    hyp_name = sys.argv[1] + '/hyp.wrd.trn'
    text_name = sys.argv[1] + '/text'
    with open(ref_name, 'r') as f:
        ref = f.readlines()
    with open(hyp_name, 'r') as f:
        hyp = f.readlines()
    error = Find_Error(ref, hyp)
    with open(text_name, 'r') as f:
        text = f.readlines()
    sampled_text = sys.argv[1] + '/train_sample.txt'
    error_text = sys.argv[1] + '/error_sample.txt'
    Sample_text(text, error, sampled_text, error_text)
    #tag = Check_Label(text, error)
    #text_refine = Refine_Text(text, tag)
    #Concatenate_Text(text_refine, error, tag, len(text))
