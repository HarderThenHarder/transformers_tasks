"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2023-03-24 20:49:03
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2023-05-06 23:34:14
FilePath: /Open-Llama/utils/train_tokenizer.py
Description: 

Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
"""
import random
from glob import glob
from datasets import load_dataset


random.seed(42)


dataset = {
    "MNBVC_news": "../data/pretrain_data/MNBVC_news/*.jsonl.zst",
    "MNBVC_qa": "../data/pretrain_data/MNBVC_qa/*.jsonl.zst",
    "MNBVC_wiki": "../data/pretrain_data/MNBVC_wiki/*.jsonl.zst",
}

paths = []
for _, pat in dataset.items():
    files = glob(pat)
    sample_num = min(len(files), 10)
    paths.extend(random.sample(files, sample_num))


dataset = load_dataset(
    "json", 
    data_files=paths, 
    split="train", 
    streaming=True
)
dataset = dataset.shuffle(seed=42)


def transform(dataset):
    for line in dataset:
        if "title" in line and "content" in line:
            yield line["title"] + "\n" + line["content"]
        else:
            yield line["text"]


data_iter = transform(dataset)

import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
spm.SentencePieceTrainer.train(
    sentence_iterator=data_iter,
    model_writer=model,
    shuffle_input_sentence=False,
    train_extremely_large_corpus=True,
    # hyperparameters of tokenizer
    max_sentence_length=16384,
    pad_id=3,
    model_type="BPE",
    vocab_size=10000,
    # split digits and fallback to byte same as Llama.
    # set split_by_unicode_script to True to avoid grouping punctuation and characters together.
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    # reserve whitespace and \n and \t etc. for code generation
    allow_whitespace_only_pieces=True,
    remove_extra_whitespaces=False,
    # Llama use identity instead of nfkc
    normalization_rule_name="nfkc",
)

# Serialize the model as file.
with open("./test_tokenizer.model", "wb") as f:
    f.write(model.getvalue())

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.decode(sp.encode("只因你太美🤗▃     \n  1")))
