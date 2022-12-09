import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


srcs, tgts = [], []
with open('/home/quert/edit_NetKu/util/summary/train_examine/train.src', 'r') as f:
    for line in f.readlines():
        srcs.append(line)

with open('/home/quert/edit_NetKu/util/summary/train_examine/train.tgt', 'r') as f:
    for line in f.readlines():
        tgts.append(line)

assert len(srcs)==len(tgts)

wrap = []
for idx in range(len(srcs)):
    idx_content = {}
    idx_content['document'] = srcs[idx]
    idx_content['summary'] = tgts[idx]
    wrap.append(idx_content)
torch.save(wrap, './train_without_trigger.pt')
