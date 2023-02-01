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

transformers classifier.

Author: pankeyu
Date: 2022/10/21
"""
import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator, get_scheduler

from rich import print
from rich.table import Table
from rich.align import Align
from rich.console import Console

from utils import convert_example
from class_metrics import ClassEvaluator
from iTrainingLogger import iSummaryWriter
from FocalLoss import FocalLoss


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-uncased', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--num_labels", default=2, type=int, help="Total classes of labels.")
parser.add_argument("--use_class_weights", default=False, type=bool, help="compute class weights with sample counts of each class.")
parser.add_argument("--max_scale_ratio", default=10.0, type=float, help="max scale ratio when use class weights.")
parser.add_argument("--loss_func", default='cross_entropy', type=str, help="choose loss function.", choices=['cross_entropy', 'focal_loss'])
parser.add_argument("--focal_loss_alpha", default=0.25, type=float, help="alpha of focal loss.")
parser.add_argument("--focal_loss_gamma", default=2.0, type=float, help="gamma of focal loss.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, metric, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                                token_type_ids=batch['token_type_ids'].to(args.device),
                                attention_mask=batch['attention_mask'].to(args.device))
            predictions = outputs.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            metric.add_batch(pred_batch=predictions, gold_batch=batch["labels"].detach().cpu().numpy().tolist())
    eval_metric = metric.compute()
    model.train()
    return eval_metric['accuracy'], eval_metric['precision'], \
            eval_metric['recall'], eval_metric['f1'], \
            eval_metric['class_metrics']


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


def train():
    reset_console()
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)

    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)

    if args.use_class_weights:                                                                   # 当出现imbalance情况时进行少样本类别系数补偿
        class_count_dict = {}
        for data in dataset['train']:
            label = data['labels']
            if label not in class_count_dict:
                class_count_dict[label] = 1
            else:
                class_count_dict[label] += 1
        class_count_dict = dict(sorted(class_count_dict.items(), key=lambda x: x[0]))
        print('[+] sample(s) count of each class: ', class_count_dict)
        
        class_weight_dict = {}
        class_count_sorted_tuple = sorted(class_count_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(class_count_sorted_tuple)):
            if i == 0:
                class_weight_dict[class_count_sorted_tuple[0][0]] = 1.0                         # 数量最多的类别权重为1.0
            else:
                scale_ratio = class_count_sorted_tuple[0][1] / class_count_sorted_tuple[i][1]
                scale_ratio = min(scale_ratio, args.max_scale_ratio)                            # 数量少多少倍，loss就scale几倍
                class_weight_dict[class_count_sorted_tuple[i][0]] = scale_ratio
        print('[+] weight(s) of each class: ', class_weight_dict)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

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
    model.to(args.device)

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

    loss_list = []
    tic_train = time.time()
    metric = ClassEvaluator()

    if args.loss_func == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_func == 'focal_loss':
        criterion = FocalLoss(
            device=args.device, 
            alpha=args.focal_loss_alpha, 
            gamma=args.focal_loss_gamma
        )
    
    global_step, best_f1 = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            token_type_ids=batch['token_type_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device))
            labels = batch['labels'].to(args.device)
            loss = criterion(outputs.logits, labels)
            if args.use_class_weights:
                labels = batch['labels'].tolist()
                weights = [class_weight_dict[label] for label in labels]
                weight = sum(weights) / len(weights)
                loss *= weight
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
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                model.save_pretrained(os.path.join(cur_save_dir))
                tokenizer.save_pretrained(os.path.join(cur_save_dir))

                acc, precision, recall, f1, class_metrics = evaluate_model(model, metric, eval_dataloader)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.record()
                
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()


if __name__ == '__main__':
    train()