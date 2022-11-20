from pytorch_lightning.accelerators import accelerator
import torch
import os
import os.path
import argparse
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from transformers import Adafactor
from longformer.sliding_chunks import pad_to_window_size
from longformer import LongformerEncoderDecoderForConditionalGeneration
from longformer import LongformerEncoderDecoderConfig
# from transformers import LEDForConditionalGeneration as LongformerEncoderDecoderForConditionalGeneration
# from transformers import LEDConfig as LongformerEncoderDecoderConfig
import pandas as pd
import pdb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from datasets import load_dataset, load_metric
from dataloader import (
    get_dataloader_summ,
    get_dataloader_pretrain,
    get_dataloader_summiter,
)
import json
from pathlib import Path


tokenizer = AutoTokenizer.from_pretrained('../PRIMER_wcep')
model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained('../PRIMER_wcep')
tokenizer.add_tokens(['<KEEP>', '<ADD>', '<SUB>'])
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained('../PRIMER_wcep/new')
tokenizer.save_pretrained('../PRIMER_wcep/new')