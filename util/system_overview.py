# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
import nltk
import random, time
import datetime
# nltk.download("stopwords")
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.metrics import classification_report
import transformers
from transformers import BartForSequenceClassification, AdamW, BartTokenizer, get_linear_schedule_with_warmup, pipeline, set_seed
from transformers import pipeline, set_seed, BartTokenizer
from datasets import load_dataset, load_metric
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from datasets import Dataset, load_metric
import datasets
import gradio as gr
import openai
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
# from vicuna_generate import *
# from convert_article import *

# Data preprocessing

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).

# Create the learning rate scheduler.

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def decode(paragraphs_needed):
    # model_ckpt = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained("tokenizer-decoder")
    # pipe = pipeline("summarization", model="bart-decoder",tokenizer=tokenizer)
    pipe = pipeline("summarization", model="hyesunyun/update-summarization-bart-large-longformer",tokenizer=tokenizer)
    contexts = [str(pipe(paragraph)) for paragraph in paragraphs_needed]
    return contexts

def split_article(article):
    # if "\c" in article.split():
    paragraphs = article.replace("\\c\\c", "\c\c").split("\\\\c\\\\c")
    pars = [par.strip() for par in paragraphs if par]
   #  else: 
   #      paragraphs = article.split("\n")
   #      pars = [par.strip() for par in paragraphs if par]
    pd.DataFrame({"paragraph": pars}).to_csv("./experiments/input_paragraphs.csv")
    return pars 

def config():
    load_dotenv()

def call_gpt(paragraph, trigger):
    openai.api_key = os.getenv("openai_apikey")
    tokenizer = BartTokenizer.from_pretrained("tokenizer-decoder")
    inputs_for_gpt = f"""
As an article writer, your task is to provide an updated paragraph in the length same as non-updated paragraph based on the given non-updated paragraph and a triggered news.
    Non-updated paragraph:
    {paragraph}

    Triggered News:
    {trigger}
        """
        # merged_with_prompts.append(merged.strip())
        # pd.DataFrame({"paragraph": merged_with_prompts}).to_csv("./experiments/paragraphs_with_prompts.csv")

    completion = openai.ChatCompletion.create(
         model = "gpt-3.5-turbo",
         messages = [
             {"role": "user", "content": inputs_for_gpt}
         ]
     )
    response = completion.choices[0].message.content
    return str(response)

def call_vicuna(paragraphs_tirgger):
    tokenizer = BartTokenizer.from_pretrained("tokenizer-decoder")
    merged_with_prompts = []
    for paragraph in paragraphs:
        merged = f"""
As an article writer, your task is to provide an updated paragraph in the length same as non-updated paragraph based on the given non-updated paragraph and a triggered news.
    Non-updated paragraph:
    {paragraph}

    Triggered News:
    {trigger}
        """
        merged_with_prompts.append(merged.strip())
        pd.DataFrame({"paragraph": merged_with_prompts}).to_csv("./experiments/paragraphs_with_prompts.csv")
    responses = vicuna_output()
    return responses

    

def main(input_article, input_trigger):
    csv_path = "./experiments/input_paragraphs.csv"
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    modified = "TRUE"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained('tokenizer-encoder')
    batch_size = 8
    model = torch.load("bart_model")
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5,
                      eps = 1e-8
                    )

    # split the input article to paragraphs in tmp csv format
    data_test = split_article(input_article)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    input_ids = []
    attention_masks = []
    for sent in data_test:
        encoded_dict = tokenizer.encode_plus(
                            text_preprocessing(sent),
                            add_special_tokens = True,
                            max_length = 1024,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    test_dataset = TensorDataset(input_ids, attention_masks)
    test_dataloader = DataLoader(
                test_dataset,
                sampler = SequentialSampler(test_dataset),
                batch_size = batch_size
            )

    # Predictions
    predictions = []
    for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            with torch.no_grad():
                output= model(b_input_ids,
                              attention_mask=b_input_mask)
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                predictions.extend(list(pred_flat))

    # Write predictions for each paragraph
    df_output = pd.DataFrame({"target": predictions}).to_csv('./experiments/classification.csv', index=False)
    if len(data_test)==1: predictions[0] = 1

    # extract ids for update-needed paragraphs (extract the idx with predicted target == 1)
    pos_ids = [idx for idx in range(len(predictions)) if predictions[idx]==1]
    neg_ids = [idx for idx in range(len(predictions)) if predictions[idx]==0]

    # feed the positive paragraphs to decoder
    paragraphs_needed = [data_test[idx] for idx in pos_ids]
    pd.DataFrame({"paragraph": paragraphs_needed}).to_csv("./experiments/paragraphs_needed.csv", index=False)

    # updated_paragraphs = decode(input_paragraph, input_trigger)
    config()
    updated_paragraphs = [call_gpt(paragraph, input_trigger) for paragraph in paragraphs_needed]
    # updated_paragraphs = call_vicuna(paragraphs_needed, input_trigger)

    # merge updated paragraphs with non-updated paragraphs
    paragraphs_merged = data_test.copy()
    for idx in range(len(pos_ids)):
        paragraphs_merged[pos_ids[idx]] = updated_paragraphs[idx]

    sep = "\n"
    updated_article = str(sep.join(paragraphs_merged))
    updated_article = updated_article.replace("[{'summary_text': '", "").replace("'}]", "").strip()
    class_res = pd.read_csv("./experiments/classification.csv")
    if class_res.target.values.all() == 0: modified="False"

    if len(data_test)==1: 
        modified="TRUE"
        updated_article = call_gpt(input_article, input_trigger)

    # combine the predictions and paragraphs into csv format file
    merged_par_pred_df = pd.DataFrame({"paragraphs": data_test, "predictions": predictions}).to_csv("./experiments/par_with_class.csv")
    # return updated_article, modified, merged_par_pred_df
    modified_in_all = str(len(paragraphs_needed)) + " / " + str(len(data_test))
    return updated_article, modified_in_all


gr.Interface(
    fn=main,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Non-updated Article", placeholder="Input the article..."
        ),
        gr.components.Textbox(
            lines=2, label="Triggered News Event", placeholder="Input the triggered news event..."
        )
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=25,
            label="Output",
        ),
        gr.inputs.Textbox(
            lines=1,
            label="#MODIFIED/ALL"
        ),
        # gr.Dataframe()
    ],
    title="Event Triggered Article Updating System",
    description="Powered by YTLee",
).queue().launch(share=True)
