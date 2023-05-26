# -*- coding: utf-8 -*-

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
from datasets import Dataset, load_metric
import datasets
# nltk.download("punkt")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, 
                               batch_size=16, device=device, 
                               column_text="article", 
                               column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):
        
        inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
                        padding="max_length", return_tensors="pt")
        
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device), 
                         length_penalty=0.8, num_beams=8, max_length=128)
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=True) 
               for s in summaries]      
        
        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
        
        
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
    score = metric.compute()
    return score

train_df = pd.read_csv("/content/drive/MyDrive/NetKu/experiments/same_secs_insert_labeled/train.csv")
train_df = train_df[["document", "summary"]]
test_df = pd.read_csv("/content/drive/MyDrive/NetKu/experiments/same_secs_insert_labeled/test.csv")
test_df = test_df[["document", "summary"]]
val_df = pd.read_csv("/content/drive/MyDrive/NetKu/experiments/same_secs_insert_labeled/val.csv")
val_df = val_df[["document", "summary"]]
train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)
val_data = Dataset.from_pandas(val_df)

rouge_metric = load_metric('rouge')

def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['document'] , max_length = 1024, truncation = True )
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length = 128, truncation = True )
        
    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }
    
train_data_pt = train_data.map(convert_examples_to_features, batched = True)
val_data_pt = val_data.map(convert_examples_to_features, batched = True)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer_args = TrainingArguments(
    output_dir='output_decoder', num_train_epochs=1, warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16
)

trainer = Trainer(model=model, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=train_data_pt,
                  eval_dataset=val_data_pt)

trainer.train()

test_data_pt = test_data.map(convert_examples_to_features, batched = True)

## Save model
model.save_pretrained("bart_decoder")

## Save tokenizer
tokenizer.save_pretrained("tuned_bart_tokenizer")

# tokenizer = AutoTokenizer.from_pretrained("tuned_bart_tokenizer")

# gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

# pipe = pipeline("summarization", model="tuned_bart_decoder",tokenizer=tokenizer)

# text = ''
# print(pipe(text))
