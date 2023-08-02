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

Multi-GPU ChatGLM Fine-tune。

Author: pankeyu
Date: 2023/03/29

Reference:
    https://github.com/mymusise/ChatGLM-Tuning
    https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
"""
import os
import copy
import time
import argparse
from functools import partial

import peft
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast

from transformers import AutoTokenizer, AutoConfig, AutoModel, default_data_collator, get_scheduler

from rich import print
from rich.table import Table
from rich.align import Align
from rich.console import Console

from utils import convert_example
from iTrainingLogger import iSummaryWriter

from accelerate import Accelerator


local_rank = int(os.environ.get("LOCAL_RANK", 0))
print('local_rank: ', local_rank)


parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_source_seq_len", default=512, type=int,help="The maximum total encoder input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--max_target_seq_len", default=512, type=int,help="The maximum total decoder input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--save_freq", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--use_lora", default=False, type=bool, help="If use LoRA.")
parser.add_argument("--use_ptuning", default=False, type=bool, help="If use P-Tuning.")
parser.add_argument("--lora_rank", default=8, type=int, help="LoRA Rank.")
parser.add_argument("--pre_seq_len", default=128, type=int, help="PTuning prefix tokens num.")
parser.add_argument("--prefix_projection", default=False, type=bool, help="Use prefix projection or not.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="Processing numbers for dataset process.")
parser.add_argument("--quantization_bit", default=None, type=int, help="Quantization bit.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def second2time(seconds: int):
    """
    将秒转换成时分秒。

    Args:
        seconds (int): _description_
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def evaluate_model(model, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in data_loader:
            loss = model(
                input_ids=batch['input_ids'].to(dtype=torch.long),
                labels=batch['labels'].to(dtype=torch.long)
            ).loss
            loss_list.append(float(loss.cpu().detach()))
    model.train()
    return sum(loss_list) / len(loss_list)


def reset_console():
    """
    重置终端，便于打印log信息。
    """
    console = Console()
    table = Table(show_footer=False)
    table.title = ("[bold not italic]:robot:[/] Config Parameters")
    table.add_column("key", no_wrap=True)
    table.add_column("value", no_wrap=True)
    
    for arg in vars(args):
        table.add_row(arg, str(getattr(args, arg)))
    
    table.caption = "You can change config in [b not dim]Source Code[/]"
    table.columns[0].style = "bright_red"
    table.columns[0].header_style = "bold bright_red"
    table.columns[1].style = "bright_green"
    table.columns[1].header_style = "bold bright_green"
    table_centered = Align.center(table)
    console.print(table_centered)


def save_model(
        model, 
        cur_save_dir: str
    ):
    """
    存储当前模型。

    Args:
        cur_save_path (str): 存储路径。
    """
    if args.use_lora:                       # merge lora params with origin model
        merged_model = copy.deepcopy(model)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(cur_save_dir)
    else:
        model.save_pretrained(cur_save_dir)


def get_optimizer_and_scheler(model, train_dataloader):
    """
    刷新optimizer和lr衰减器。
    如果设置了auto_da，则每做一次自动数据增强都需要重置学习率。

    Args:
        model (_type_): _description_
        train_dataloader (_type_): _description_

    Returns:
        _type_: _description_
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler, max_train_steps


def main():
    if local_rank == 0:
        reset_console()

    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", 
        trust_remote_code=True
    )

    config = AutoConfig.from_pretrained(
        "THUDM/chatglm-6b", 
        trust_remote_code=True
    )
    if args.use_ptuning:
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection

    model = AutoModel.from_pretrained(
        "THUDM/chatglm-6b",
        config=config,
        trust_remote_code=True
    )

    if args.quantization_bit is not None:
        print(f"Quantized to {args.quantization_bit} bit")
        model = model.quantize(args.quantization_bit)

    model = model.half()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False
    if args.use_ptuning:
        model.transformer.prefix_encoder.float()

    if args.use_lora:
        model.lm_head = CastOutputToFloat(model.lm_head)
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = peft.get_peft_model(model, peft_config)

    # load dataset
    dataset = load_dataset('text', data_files={
        'train': args.train_path,
        'dev': args.dev_path
    })
    if local_rank == 0:
        print(dataset)

    convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_source_seq_len=args.max_source_seq_len,
        max_target_seq_len=args.max_target_seq_len,
    )
    
    dataset = dataset.map(
        convert_func, 
        batched=True,
        num_proc=args.preprocessing_num_workers
    )

    train_dataset = dataset["train"]
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=default_data_collator
    )

    eval_dataset = dataset["dev"]
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=default_data_collator, 
        batch_size=args.batch_size
    )

    optimizer, lr_scheduler, max_train_steps = get_optimizer_and_scheler(model, train_dataloader)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, 
        optimizer, 
        train_dataloader, 
        lr_scheduler
    )

    loss_list = []
    tic_train = time.time()
    total_start = time.time()
    global_step, best_eval_loss = 0, float('inf')
    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model(
                input_ids=batch['input_ids'].to(dtype=torch.long),
                labels=batch['labels'].to(dtype=torch.long)
            ).loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                
                if local_rank == 0:
                    writer.add_scalar('train/train_loss', loss_avg, global_step)
                    print("global step %d ( %02.2f%% ) , epoch: %d, loss: %.5f, speed: %.2f step/s, ETA: %s"
                            % (
                        global_step, 
                        global_step / max_train_steps * 100, 
                        epoch, 
                        loss_avg, 
                        args.logging_steps / time_diff,
                        second2time(int(max_train_steps - global_step) / (args.logging_steps / time_diff))
                    ))
                    tic_train = time.time()

            if global_step % args.save_freq == 0 and local_rank == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                save_model(model.module, cur_save_dir)
                tokenizer.save_pretrained(cur_save_dir)
                print(f'Model has saved at {cur_save_dir}.')

                eval_loss = evaluate_model(model, eval_dataloader)
                writer.add_scalar('eval/evaluate_loss', eval_loss, global_step)
                writer.record()
            
                print("Evaluation Loss: %.5f" % (eval_loss))
                if eval_loss < best_eval_loss:
                    print(
                        f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}"
                    )
                    best_eval_loss = eval_loss
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    save_model(model.module, cur_save_dir)
                    tokenizer.save_pretrained(cur_save_dir)
                    print(f'Best model has saved at {cur_save_dir}.')

                tic_train = time.time()

    if local_rank == 0:
        used_time = second2time(int(time.time() - total_start))
        print(f'[done]Used {used_time}.')


if __name__ == "__main__":
    main()