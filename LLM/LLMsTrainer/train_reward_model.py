# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

Training Reward Model.

Author: pankeyu
Date: 2023/08/01
"""
import yaml
import argparse

from rich import print
from accelerate import Accelerator

from transformers import (
    AutoTokenizer, 
    LlamaTokenizer,
    AutoModelForSequenceClassification
)
from dataset.reward_model_dataset import construct_dataloader
from solver.reward_model_trainer import RewardModelTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--train_config", default=None, type=str, help="Train config path.")
parser.add_argument("--model_config", default=None, type=str, help="Model config, when you specify ckpt, this param won't be used.")
args = parser.parse_args()


with open(args.train_config, "r", encoding="utf-8") as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)
    print('=' * 20 + ' Current Config ' + '=' * 20)
    print(config)
    

def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get(
            "gradient_accumulation_steps", 1
        )
    )

    if (
        'llama' in config["data"]["tokenizer_path"] 
        or
        'Llama' in config["data"]["tokenizer_path"] 
    ):
        tokenizer = LlamaTokenizer.from_pretrained(
            config["data"]["tokenizer_path"]
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["data"]["tokenizer_path"], 
            trust_remote_code=True
        )

    assert tokenizer.pad_token, \
        'Detect pad_token is None, you should specify "pad_token" in tokenizer config.'

    assert tokenizer.eos_token, \
        'Detect eos_token is None, you should specify "eos_token" in tokenizer config.'

    model = AutoModelForSequenceClassification.from_pretrained(
        config["train"]["ckpt"], 
        num_labels=1,
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    if config["train"].get("gradient_checkpointing_enable", False):
        model.gradient_checkpointing_enable()
    
    if config["train"].get("downscale_weight", False):
        model.score.weight.data *= 0.1
    
    dataloader = construct_dataloader(
        config['data'],
        tokenizer
    )

    trainer = RewardModelTrainer(
        config, 
        model, 
        dataloader, 
        tokenizer, 
        accelerator
    )
    trainer.train()
    

if __name__ == '__main__':
    main()
