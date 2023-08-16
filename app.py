# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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
import pyperclip
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
    tokenizer = AutoTokenizer.from_pretrained("theQuert/NetKUp-tokenzier")
    # pipe = pipeline("summarization", model="bart-decoder",tokenizer=tokenizer)
    pipe = pipeline("summarization", model="hyesunyun/update-summarization-bart-large-longformer",tokenizer=tokenizer)
    contexts = [str(pipe(paragraph)) for paragraph in paragraphs_needed]
    return contexts

def split_article(article, trigger):
    if len(article.split("\\n ")) > len(article.split("\\\\c\\\\c")):
        paragraphs = article.split("\\n ")
    else:
        paragraphs = article.split("\\\\c\\\\c")
    pars = [str(par) + " -- " + str(trigger) for par in paragraphs]
    # pd.DataFrame({"paragraphs": paragraphs}).to_csv("./util/experiments/check_par.csv")
    format_pars = [par for par in paragraphs]
    formatted_input = "\n".join(format_pars)
    return pars, formatted_input 

def config():
    load_dotenv()

def call_gpt(paragraph, trigger):
    # openai.api_key = os.environ.get("GPT-API")
    tokenizer = BartTokenizer.from_pretrained("theQuert/NetKUp-tokenzier")
    inputs_for_gpt = f"""
s an article writer, your task is to provide an updated paragraph in the length same as non-updated paragraph based on the given non-updated paragraph and a triggered news.Remember, the length of updated paragraph is restricted into a single paragraph.
    Non-updated paragraph:
    {paragraph}

    Triggered News:
    {trigger}
        """
    completion = openai.ChatCompletion.create(
         model = "gpt-3.5-turbo",
         messages = [
             {"role": "user", "content": inputs_for_gpt}
         ]
     )
    response = completion.choices[0].message.content
    if "<"+response.split("<")[-1].strip() == "<"+paragraph.split("<")[-1].strip(): response = response 
    else: response = response + " <"+paragraph.split("<")[-1].strip()
    return str(response)

def call_vicuna(paragraphs_tirgger):
    tokenizer = BartTokenizer.from_pretrained("theQuert/NetKUp-tokenzier")
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
        pd.DataFrame({"paragraph": merged_with_prompts}).to_csv("./util/experiments/paragraphs_with_prompts.csv")
    responses = vicuna_output()
    return responses

    
def main(input_article, input_trigger):
    paths = [".util/experiments/input_paragraphs.csv",
             "./util/experiments/formatted_input.txt",
             "./util/experiments/updated_article.txt",
             "./util/experiments/paragraphs_needed.txt",
             "./util/experiments/updated_paragraphs.txt",
             "./util/experiments/paragraphs_with_prompts.csv",
             "./util/experiments/classification.csv",
             "./util/experiments/paragraphs_needed.csv",
             "./util/experiments/par_with_class.csv",
             ]
    for path in paths: 
        try:
            if os.path.isfile(path): os.remove(path)
        except: continue 
    modified = "TRUE"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained('theQuert/NetKUp-tokenzier')
    batch_size = 8
    model = torch.load("./util/bart_model")
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5,
                      eps = 1e-8
                    )

    # split the input article to paragraphs in tmp csv format
    data_test, formatted_input = split_article(input_article, input_trigger)
    with open("./util/experiments/formatted_input.txt", "w") as f:
        f.write(formatted_input)

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
                            max_length = 600,
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
    df_output = pd.DataFrame({"target": predictions}).to_csv('./util/experiments/classification.csv', index=False)
    if len(data_test)==1: predictions[0] = 1

    # extract ids for update-needed paragraphs (extract the idx with predicted target == 1)
    pos_ids = [idx for idx in range(len(predictions)) if predictions[idx]==1]
    neg_ids = [idx for idx in range(len(predictions)) if predictions[idx]==0]

    # feed the positive paragraphs to decoder
    paragraphs_needed = [data_test[idx] for idx in pos_ids]
    paragraphs_needed = [par.split(" -- ")[0].replace("[ADD]", "") for par in paragraphs_needed]
    pd.DataFrame({"paragraph": paragraphs_needed}).to_csv("./util/experiments/paragraphs_needed.csv", index=False)
    paragraphs_needed_str = "\n\n".join(paragraphs_needed)
    # paragraphs_needed_str = paragraphs_needed_str.replace("Updated Paragraph:\n", "")
    with open("./util/experiments/paragraphs_needed.txt", "w") as f:
        f.write(paragraphs_needed_str)

    # updated_paragraphs = decode(input_paragraph, input_trigger)
    # updated_paragraphs = call_vicuna(paragraphs_needed, input_trigger)
    config()
    updated_paragraphs = [call_gpt(paragraph, input_trigger) for paragraph in paragraphs_needed]
    updated_paragraphs_str = "\n\n".join(updated_paragraphs)
    updated_paragraphs_str = updated_paragraphs_str.replace("Updated Paragraph:\n", "")
    with open("./util/experiments/updated_paragraphs.txt", "w") as f:
        f.write(updated_paragraphs_str)

    # merge updated paragraphs with non-updated paragraphs
    paragraphs_merged = data_test.copy()
    paragraphs_merged = [str(par).split(" -- ")[0] for par in paragraphs_merged]
    for idx in range(len(pos_ids)):
        paragraphs_merged[pos_ids[idx]] = updated_paragraphs[idx]

    sep = "\n"
    # paragarphs_merged = ["".join(par.split(" -- ")[:-1]) for par in paragraphs_merged]
    updated_article = str(sep.join(paragraphs_merged))
    updated_article = updated_article.replace("[{'summary_text': '", "").replace("'}]", "").strip()
    class_res = pd.read_csv("./util/experiments/classification.csv")
    if class_res.target.values.all() == 0: modified="False"

    if len(data_test)==1: 
        modified="TRUE"
        updated_article = call_gpt(input_article, input_trigger)
    with open("./util/experiments/updated_article.txt", "w") as f:
        f.write(updated_article)

    # combine the predictions and paragraphs into csv format file
    merged_par_pred_df = pd.DataFrame({"paragraphs": data_test, "predictions": predictions}).to_csv("./util/experiments/par_with_class.csv")
    # return updated_article, modified, merged_par_pred_df
    modified_in_all = str(len(paragraphs_needed)) + " / " + str(len(data_test))
    return updated_article, modified_in_all

def copy_to_clipboard(t):
    with open("./util/experiments/updated_article.txt", "r") as f:
        t = f.read()
        pyperclip.copy(t)

def compare_versions():
    with open("./util/experiments/paragraphs_needed.txt", "r") as f:
        old = f.read()
        old = old.replace("[ADD]", "")
    with open("./util/experiments/updated_paragraphs.txt", "r") as f:
        new = f.read()
        new = new.replace("[ADD]", "")
    return old, new

with open("./examples/non_update.txt", "r") as f:
    exin_1 = f.read()
with open("./examples/trigger.txt", "r") as f:
    trigger_1 = f.read()
with open("./examples/non_update_2.txt", "r") as f:
    exin_2 = f.read()
with open("./examples/trigger_2.txt", "r") as f:
    trigger_2 = f.read()

with gr.Blocks() as demo:
    gr.HTML("""<div style="text-align: center; max-width: 700px; margin: 0 auto;">
            <div
            style="
                display: inline-flex;
                align-items: center;
                gap: 0.8rem;
                font-size: 1.75rem;
            "
            >
            <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
                Event Triggered Article Updating System
            </h1>
            </div>"""
        )
    with gr.Tab("Article Updating"):
        input_1 = gr.Textbox(label="Non-updated Article", lines=2, placeholder="Input the contexts...")
        input_2 = gr.Textbox(label="Triggered News Event", lines=1, placeholder="Input the triggered news event...") 
        btn = gr.Button(value="Submit")
        with gr.Row():
            output_1 = gr.Textbox(label="Updated Article", lines=5)
            output_2 = gr.Textbox(label="#MODIFIED / #ALL")
        btn.click(fn=main, inputs=[input_1, input_2], outputs=[output_1, output_2])
        btn_copy = gr.Button(value="Copy Updated Article to Clipboard")
        btn_copy.click(fn=copy_to_clipboard, inputs=[output_1], outputs=[])
        gr.Markdown("## Input Examples")
        gr.Markdown("### There are 2 examples below, click them to test inputs automatically!")
        gr.Examples(
            examples=[[exin_1, trigger_1], [exin_2, trigger_2]],
            fn=main,
            inputs=[input_1, input_2],
            outputs=[output_1, output_2],
            # cache_examples=True,
            # run_on_click=True,
                ),
        com_1_value, com_2_value = "Pls finish article updating, then click the button above", "Pls finish article updating, then click the button above."
    with gr.Tab("Compare between versions"):
        btn_com = gr.Button(value="Differences Highlighting")
        with gr.Row():
            com_1 = gr.Textbox(label="Non-update Article", value=com_1_value, lines=15)
            com_2 = gr.Textbox(label="Updated Article", value=com_2_value, lines=15)
        btn_com.click(fn=compare_versions, inputs=[], outputs=[com_1, com_2])
    gr.HTML("""
            <div align="center">
                <p>
                Demo by 🤗 <a href="https://github.com/thequert" target="_blank"><b>Yu-Ting Lee</b></a>
                </p>
            </div>
            <div align="center">
                <p>
                Supported by <a href="https://www.nccu.edu.tw/"><b>National Chengchi University</a></b> & <a href="https://www.sinica.edu.tw/"><b>Academia Sinica</b></a>
                </p>
            </div>
        """
        )

demo.launch(server_name="0.0.0.0", server_port=7840)

