from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
nltk.download("punkt")
nltk.download("stopwords")

# Import the positive paragraphs
test_pos = pd.read_csv("../dataset/same_secs_insert_labeled/pos_classification.csv")
pos_instances = test_pos.document.values

# Load model
tokenizer = AutoTokenizer.from_pretrained("./tokenizer-decoder/")
pipe = pipeline("summarization", model="./bart-decoder/", tokenizer=tokenizer)

# Generate summaries given positive paragraphs
outputs = []
for idx in tqdm(range(len(pos_instances))):
    updated_par = pipe(pos_instances[idx])
    outputs.append(updated_par)

pd.DataFrame({"paragraph": outputs}).to_csv("./generated_paragraphs.csv")
