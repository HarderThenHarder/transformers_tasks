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

参照 OpenLlma 改写训练文件，支持训练OpenLLAMA以外的其他模型，包含MPT等。

Authors: pankeyu
Date: 2023/05/26

Reference Authors: s-JoL(sl12160010@gmail.com)
Reference repo: Open-Llama
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import argparse

from rich import print
from accelerate import Accelerator
from torch.utils.data import DataLoader

from peft import LoraConfig, TaskType, get_peft_model
from datasets.distributed import split_dataset_by_node
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

from dataset.dataset import construct_dataset
from solver.trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("--train_config", default=None, type=str, help="Train config path.")
parser.add_argument("--model_config", default=None, type=str, help="Model config, when you specify ckpt, this param won't be used.")
args = parser.parse_args()


with open(args.train_config, "r", encoding="utf-8") as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)
    print('=' * 20 + ' Current Config ' + '=' * 20)
    print(config)


def main():
    assert args.model_config or config["train"]["ckpt"], \
        'You must specify model_config by passing --model_config or a pretrained model path by setting `ckpt` in --train_config.'

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))                    # 将timeout阈值从 30min 设置为 1h
    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get(
            "gradient_accumulation_steps", 1
        ),
        kwargs_handlers=[kwargs]
    )
    
    if (
        'llama' in config["data"]["tokenizer_path"] 
        or
        'Llama' in config["data"]["tokenizer_path"] 
    ):
        tokenizer = LlamaTokenizer.from_pretrained(config["data"]["tokenizer_path"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_path"])

    assert tokenizer.pad_token, \
        'Detect pad_token is None, you should specify "pad_token" in tokenizer config.'

    assert tokenizer.eos_token, \
        'Detect eos_token is None, you should specify "eos_token" in tokenizer config.'

    test_tokens = tokenizer('this is a test sentence')['input_ids']                    # 测试tokenizer是否会自动添加 eos token，
    config['data']['added_eos_token'] = test_tokens[-1] == tokenizer.eos_token_id      # 用于后续判断是否自动为句子添加 eos token

    data_config = config["data"]
    if data_config.get("split_by_shard", False):
        train_dataset = construct_dataset(
            data_config, 
            tokenizer, 
            world_size=accelerator.num_processes
        )
    else:
        train_dataset = construct_dataset(data_config, tokenizer)
        
    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["train_batch_size"],
        num_workers=config["train"]["train_num_workers"],
        prefetch_factor=config["train"].get("prefetch_factor", 2),
        pin_memory=True,
    )

    model_config = AutoConfig.from_pretrained(                                      
        args.model_config,
        trust_remote_code=True
    )
    model_config.pad_token_id = tokenizer.pad_token_id
    print('tokenizer vocab_size: ', len(tokenizer))

    if config["train"]["ckpt"] is not None:                                          # 在已有的预训练模型上进行continue training
        print("Loading ckpt from: {}".format(config["train"]["ckpt"]))
        raw_model = AutoModelForCausalLM.from_pretrained(
            config["train"]["ckpt"], 
            trust_remote_code=True
        )
        
        if model_config.vocab_size != len(tokenizer):
            resize_flag = config["train"].get("resize_model_vocab_size", True)
            if not resize_flag:                                                       # 当 tokenizer 和 model 的 vocab_size 不一致，且 train_config 中指定不进行 resize 操作时，抛出一个 warning
                print(f'[Vocab Warning] current model_vocab_size is `{model_config.vocab_size}`, which is not equal to tokenizer vocab_size `{len(tokenizer) }`, but `resize_model_vocab_size` is not in train_config or set as False.')
            else:
                raw_model.resize_token_embeddings(len(tokenizer))
                print(f"Resize model vocab_size from {model_config.vocab_size} to {len(tokenizer)}.")
    else:
        model_config.vocab_size = len(tokenizer)                                      # 从头开始训练
        raw_model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=True
        )

    if config["train"].get("train_token_embeddings_only", False):                     # 是否只训练 token embedding
        print('[Train Token Embedding Only] Start to freeze model...')
        token_embedding_layer_name = config["train"].get("token_embedding_layer_name", "")
        assert token_embedding_layer_name, "[Train Token Embedding Only] Can not find `token_embedding_layer_name` in train config, or you can set `train_token_embeddings_only` to False."
        
        from utils.model_freeze import freeze_model_exclude_token_embeddings
        freeze_model_exclude_token_embeddings(
            raw_model,
            token_embedding_layer_name
        )
        print('[Train Token Embedding Only] Initialized Model.')

    if config["train"].get("use_lora", False):
        if hasattr(raw_model, "enable_input_require_grads"):
            raw_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            raw_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        target_modules = config['train'].get('target_modules', [])
        assert target_modules, 'You must specify `target_modules` since you want to use LoRA.'
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        raw_model = get_peft_model(raw_model, peft_config)
        raw_model.print_trainable_parameters()
    
    if config["train"].get("gradient_checkpointing_enable", False):
        raw_model.gradient_checkpointing_enable()
    
    trainer = Trainer(
        config, 
        raw_model, 
        train_loader, 
        tokenizer, 
        accelerator
    )
    trainer.train()


if __name__ == "__main__":
    main()
