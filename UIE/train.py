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

torch 版本 UIE fintuning 脚本。

Author: pankeyu
Date: 2022/09/06
"""
import os
import time
import json
import random
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator, get_scheduler

from rich.table import Table
from rich.align import Align
from rich.console import Console
from rich import print

from Augmenter import Augmenter
from inference import inference
from metrics import SpanEvaluator
from model import UIE, convert_example
from utils import download_pretrained_model
from iTrainingLogger import iSummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", default='uie-base-zh', type=str, choices=['uie-base-zh'], help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoint", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=300, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--auto_neg_rate", default=0.5, type=float, help="Auto negative samples generated ratio.")
parser.add_argument("--auto_pos_rate", default=0.5, type=float, help="Auto positive samples generated ratio.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--auto_da_epoch", default=0, type=int, required=False, help="auto add positive/negative samples policy frequency.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate(model, metric, data_loader, global_step):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in data_loader:
            start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                            token_type_ids=batch['token_type_ids'].to(args.device),
                                            attention_mask=batch['attention_mask'].to(args.device))
            start_ids = batch['start_ids'].to(torch.float32).detach().numpy()
            end_ids = batch['end_ids'].to(torch.float32).detach().numpy()
            num_correct, num_infer, num_label = metric.compute(start_prob.cpu().detach().numpy(), 
                                                                end_prob.cpu().detach().numpy(), 
                                                                start_ids, 
                                                                end_ids)
            metric.update(num_correct, num_infer, num_label)
        
        precision, recall, f1 = metric.accumulate()
        writer.add_scalar('eval-precision', precision, global_step)
        writer.add_scalar('eval-recall', recall, global_step)
        writer.add_scalar('eval-f1', f1, global_step)
        writer.record()
    model.train()
    return precision, recall, f1

def auto_add_samples(
    model, 
    tokenizer,
    epoch: int,
    convert_func,
    ) -> DataLoader:
    """
    根据模型当前学习的效果，自动添加正例/负例。

    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        epoch (int): 当前步数
        convert_func (_type_): 数据集map函数
    
    Returns:
        加入了负例后的 train_dataloader
    """
    model.eval()
    with torch.no_grad():
        dataset_path = os.path.dirname(args.train_path)
        train_dataset_with_new_samples_added = os.path.join(dataset_path, 'new_train.txt')      # 加入正/负例后的训练数据集存放地址
        da_sample_details_path = os.path.join(dataset_path, 'auto_da_details')                  # 自动增加负例的详细信息文件存放地址
        if not os.path.exists(da_sample_details_path):
            os.makedirs(da_sample_details_path)
        
        neg_sample_file = os.path.join(da_sample_details_path, f'neg_samples_{epoch}.txt')      # 生成的负例数据存放位置
        neg_details_file = os.path.join(da_sample_details_path, f'neg_details_{epoch}.log')     # 生成的详情存放位置
        pos_sample_file = os.path.join(da_sample_details_path, f'pos_samples_{epoch}.txt')      # 生成的正例数据存放位置

        Augmenter.auto_add_uie_relation_negative_samples(
            model,
            tokenizer,
            samples=[args.train_path],
            inference_func=inference,
            negative_samples_file=neg_sample_file,
            details_file=neg_details_file,
            device=args.device,
            max_seq_len=args.max_seq_len       
        )

        Augmenter.auto_add_uie_relation_positive_samples(
            samples=[args.train_path],
            positive_samples_file=pos_sample_file
        )

        generated_negative_samples = [line.strip() for line in open(neg_sample_file, 'r', encoding='utf8').readlines()]
        generated_positive_samples = [line.strip() for line in open(pos_sample_file, 'r', encoding='utf8').readlines()]
        train_samples = [eval(line.strip()) for line in open(args.train_path, 'r', encoding='utf8').readlines()]
        train_positive_samples, train_negaitive_samples = [], []
        for train_sample in train_samples:
            if train_sample['result_list']:
                train_positive_samples.append(json.dumps(train_sample, ensure_ascii=False))                   # 保留训练数据集中的正例
            else:
                train_negaitive_samples.append(json.dumps(train_sample, ensure_ascii=False))                  # 保存训练数据集中的负例
        
        # * 添加正/负例
        negaitve_samples_generated_sample_num = int(len(generated_negative_samples) * args.auto_neg_rate)     # 随机采样等比例的新生成负例数据
        positive_samples_generated_sample_num = int(len(generated_positive_samples) * args.auto_pos_rate)     # 随机采样等比例的新生成正例数据

        # * 新数据集混合
        new_train_samples = train_positive_samples + train_negaitive_samples + \
                random.sample(generated_positive_samples, k=positive_samples_generated_sample_num) + \
                random.sample(generated_negative_samples, k=negaitve_samples_generated_sample_num)

        with open(train_dataset_with_new_samples_added, 'w', encoding='utf8') as f:                            # 保存新的训练数据集
            for line in new_train_samples:
                f.write(f'{line}\n')
        args.train_path = train_dataset_with_new_samples_added                                                 # 替换args中训练数据集为最新的训练集

        train_dataset = load_dataset('text', data_files={'train': args.train_path})["train"]
        train_dataset = train_dataset.map(convert_func, batched=True)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    
    model.train()
    return train_dataloader


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
    num_train_epochs = args.auto_da_epoch if args.auto_da_epoch > 0 else args.num_train_epochs
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler


def train():
    reset_console()
    if not os.path.exists(args.pretrained_model):
        download_pretrained_model(args.pretrained_model)
    model = torch.load(os.path.join(args.pretrained_model, 'pytorch_model.bin'))        # 加载预训练好的UIE模型，模型结构见：model.UIE()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)                    # 加载tokenizer，ERNIE 3.0
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)
    optimizer, lr_scheduler = get_optimizer_and_scheler(model, train_dataloader)

    loss_list = []
    tic_train = time.time()
    metric = SpanEvaluator()
    criterion = torch.nn.BCELoss()
    global_step, best_f1 = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                        token_type_ids=batch['token_type_ids'].to(args.device),
                                        attention_mask=batch['attention_mask'].to(args.device))
            start_ids = batch['start_ids'].to(torch.float32).to(args.device)
            end_ids = batch['end_ids'].to(torch.float32).to(args.device)
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)

                precision, recall, f1 = evaluate(model, metric, eval_dataloader, global_step)
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()
        
        if args.auto_da_epoch > 0 and epoch % args.auto_da_epoch == 0:
            train_dataloader = auto_add_samples(
                model, 
                tokenizer,
                epoch,
                convert_func
            )
            model = torch.load(os.path.join(args.pretrained_model, 'pytorch_model.bin'))        # 重新加载预训练模型
            model.to(args.device)
            optimizer, lr_scheduler = get_optimizer_and_scheler(model, train_dataloader)


if __name__ == '__main__':
    train()
