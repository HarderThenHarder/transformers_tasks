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

使用T5进行中文问答任务训练，数据集使用百度开源中文问答数据集。

Author: pankeyu
Date: 2023/01/04
"""
import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, default_data_collator, get_scheduler

from utils import convert_example
from bleu_metrics import BLEU
from iTrainingLogger import iSummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", default='uer/t5-base-chinese-cluecorpussmall', type=str, help="backbone of encoder.")
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
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--num_labels", default=2, type=int, help="Total classes of labels.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    bleu_evaluators = [BLEU(n_size=i+1) for i in range(4)]
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            outputs = model.generate(input_ids=batch['input_ids'].to(args.device))                  # (batch, seq_len)
            for prediction, reference in zip(outputs.cpu().numpy(), batch['labels'].numpy()):
                for bleu_evaluator in bleu_evaluators:
                    bleu_evaluator.add_instance(prediction=prediction, references=[reference])
    model.train()
    return [bleu.compute() for bleu in bleu_evaluators]


def train():
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.eos_token = tokenizer.sep_token                               # 因为中文用的是Bert的Tokenizer，所以需要
    tokenizer.bos_token = tokenizer.cls_token                               # 手动指定BOS Token和EOS Token
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_source_seq_len=args.max_source_seq_len,
        max_target_seq_len=args.max_target_seq_len,
    )
    dataset = dataset.map(convert_func, batched=True)
    
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
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
    global_step, best_bleu4 = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                decoder_input_ids= batch['decoder_input_ids'].to(args.device),
                labels=batch['labels'].to(args.device)
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
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                model.save_pretrained(os.path.join(cur_save_dir))
                tokenizer.save_pretrained(os.path.join(cur_save_dir))

                bleu1, bleu2, bleu3, bleu4 = evaluate_model(model, eval_dataloader)
                writer.add_scalar('eval/bleu-size-1', bleu1, global_step)
                writer.add_scalar('eval/bleu-size-2', bleu2, global_step)
                writer.add_scalar('eval/bleu-size-3', bleu3, global_step)
                writer.add_scalar('eval/bleu-size-4', bleu4, global_step)
                writer.record()
                
                print("Evaluation bleu4: %.5f" % (bleu4))
                if bleu4 > best_bleu4:
                    print(
                        f"best BLEU-4 performence has been updated: {best_bleu4:.5f} --> {bleu4:.5f}"
                    )
                    best_bleu4 = bleu4
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()


if __name__ == '__main__':
    from rich import print
    train()