import torch
import spacy
import difflib
import itertools
import re
import json
import string
import pycountry
import pandas as pd
import numpy as np

train_pt = torch.load('../../edit_NetKu/util/test_labeled.pt')
summ_pt = torch.load('../../NetKu/full_content/test.pt')

def pars_and_calculatetokens(instance):
    pars = instance.split('\n\n')
    rec_all = []
    for par in pars:
        num_add, num_sub = 0, 0
        par_rec = []
        for token in par.split():
            if token==['ADD']: num_add+=1
            elif token==['SUB']: num_sub+=1
            num_edits = num_add + num_sub
            par_rec.append(num_edits)
        rec_all.append(num_edits)
    return rec_all 

# for idx in range(len(train_pt)):
#     instance = train_pt[idx]['content']
#     rec = pars_and_calculatetokens(instance)

wrap = []
for idx in range(len(train_pt)):
    idx_content = {}
    idx_content['document'] = train_pt[idx]['content'].replace('#####', '.\n\n')
    idx_content['summary'] = summ_pt[idx]['summary']
    wrap.append(idx_content)
torch.save(wrap, './test_relabeled.pt')