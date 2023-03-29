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

ChatGLM Fine-tune。

Author: pankeyu
Date: 2023/03/26

Reference:
    https://github.com/mymusise/ChatGLM-Tuning
"""
import os
import time
import argparse
from functools import partial

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast

from transformers import AutoTokenizer, default_data_collator, get_scheduler
from modeling_chatglm import ChatGLMForConditionalGeneration

from rich import print
from rich.table import Table
from rich.align import Align
from rich.console import Console

from peft import get_peft_model, LoraConfig, TaskType
from utils import convert_example
from iTrainingLogger import iSummaryWriter


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
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--save_freq", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--lora_rank", default=8, type=int, help="LoRA Rank.")
args = parser.parse_args()


writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)


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
            with autocast():
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=args.device),
                    attention_mask=batch['attention_mask'].to(dtype=torch.bool, device=args.device),
                    position_ids= batch['position_ids'].to(dtype=torch.long, device=args.device),
                    labels=batch['labels'].to(dtype=torch.long, device=args.device)
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
    reset_console()
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "THUDM/chatglm-6b",
        load_in_8bit=False, 
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.to(args.device)

    # load dataset
    dataset = load_dataset('text', data_files={
        'train': args.train_path,
        'dev': args.dev_path
    })
    print(dataset)

    convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_source_seq_len=args.max_source_seq_len,
        max_target_seq_len=args.max_target_seq_len,
    )
    dataset = dataset.map(convert_func, batched=True)

    train_dataset = dataset["train"]
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=default_data_collator, 
        batch_size=args.batch_size
    )

    eval_dataset = dataset["dev"]
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=default_data_collator, 
        batch_size=args.batch_size
    )

    optimizer, lr_scheduler, max_train_steps = get_optimizer_and_scheler(model, train_dataloader)

    loss_list = []
    tic_train = time.time()
    global_step, best_eval_loss = 0, float('inf')
    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=args.device),
                    attention_mask=batch['attention_mask'].to(dtype=torch.bool, device=args.device),
                    position_ids= batch['position_ids'].to(dtype=torch.long, device=args.device),
                    labels=batch['labels'].to(dtype=torch.long, device=args.device)
                )
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d (%.2f%%) , epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, global_step / max_train_steps * 100, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.save_freq == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                save_tunable_parameters(model, os.path.join(cur_save_dir, "chatglm-lora.pt"))
                print(f'Model has saved at {os.path.join(cur_save_dir, "chatglm-lora.pt")}.')

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
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    save_tunable_parameters(model, os.path.join(cur_save_dir, "chatglm-lora.pt"))
                    print(f'Best model has saved at {os.path.join(cur_save_dir, "chatglm-lora.pt")}.')
                tic_train = time.time()

    print(f'done.')


if __name__ == "__main__":
    main()
