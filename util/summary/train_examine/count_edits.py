import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_pt = torch.load('./train_labeled_without_trigger.pt')

edits = []
for idx in range(len(train_pt)):
    tokens = train_pt[idx]['content']
    edit = 0
    for token in tokens:
        if token in ['[ADD]', '[SUB]', '[RM]']:
            edit += 1
    edits.append(edit)

print(f'The min edits actions is {np.min(edits)}, max is {np.max(edits)}, mean is {np.mean(edits)}, median is {np.median(edits)}'
