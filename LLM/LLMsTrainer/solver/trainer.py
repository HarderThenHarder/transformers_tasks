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

参照 OpenLlama 改写训练文件。

Authors: pankeyu
Date: 2023/05/25

Reference Authors: s-JoL(sl12160010@gmail.com)
Reference repo: Open-Llama
"""
import os
import time
import json
import shutil
import logging

import torch
from tqdm import tqdm
from torchinfo import summary
from deepspeed.ops.adam import FusedAdam
from transformers import get_cosine_schedule_with_warmup

from solver.evaluator import *
from iTrainingLogger import iSummaryWriter


class Trainer:
    
    def __init__(
            self, 
            config, 
            raw_model, 
            train_loader, 
            tokenizer, 
            accelerator
        ):
        self.config = config
        self.raw_model = raw_model
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.train_and_eval = config["train"].get("train_and_eval", False)
        self.gradient_accumulation_steps = config["train"].get(
            "gradient_accumulation_steps", 1
        )
        self.lr_scheduler_factor = (
            accelerator.num_processes / accelerator.gradient_accumulation_steps
        )
        self.log_interval = (
            self.config["log_interval"] * accelerator.gradient_accumulation_steps
        )
        self.eval_interval = (
            self.config["eval_interval"] * accelerator.gradient_accumulation_steps
        )
        self.save_interval = (
            self.config["save_interval"] * accelerator.gradient_accumulation_steps
        )
        self.work_dir = self.config["work_dir"]
        self.all_eval_methods = ["nll", "generation_eval", "single_choice_eval"]
        if accelerator.is_main_process:
            self.writer = iSummaryWriter(
                config['train'].get('img_log_dir', 'log'),
                config['train'].get('img_log_name', 'performance')
            )

    def get_model_info(self):
        with torch.no_grad():
            summary(
                self.raw_model.cuda(),
                input_data=torch.ones(1, 64, dtype=torch.int64).cuda(),
            )

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        
        if self.config["train"].get("use_lora", False):
            optimizer_grouped_parameters = self.raw_model.parameters()
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                        and p.requires_grad
                    ],
                    "weight_decay": self.config["train"]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if any(nd in n for nd in no_decay)
                        and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        
        self.optim = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.config["train"]["lr"],
            betas=(0.9, 0.95),
        )

    def get_lr_scheduler(self):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=int(
                self.config["train"]["num_warmup_steps"]
                * self.lr_scheduler_factor
            ),
            num_training_steps=int(
                self.config["train"]["num_training_steps"]
                * self.lr_scheduler_factor
            )
        )

    def prepare(self):
        """
        用于 accelerate 更新。
        """
        (
            _,
            self.model,
            self.optim,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_loader, 
            self.raw_model, 
            self.optim, 
            self.scheduler
        )
        self.optim.zero_grad()
        self.global_step = 0
        
        try:
            self.accelerator.load_state(self.work_dir)
            self.global_step = self.scheduler.scheduler._step_count - 1
            self.global_step = self.global_step // self.accelerator.num_processes
            logging.warning("Restored ckpt from {}".format(self.work_dir))
        except:
            logging.warning("No ckpt found in {}".format(self.work_dir))
        
        if self.global_step > 0:
            skip_steps = self.global_step * self.gradient_accumulation_steps
            logging.warning("Skiped {} steps.".format(skip_steps))
            self.train_loader_skipped = self.accelerator.skip_first_batches(
                self.train_loader, 
                num_batches=skip_steps
            )
        else:
            self.train_loader_skipped = self.train_loader
        
        self.accelerator.wait_for_everyone()
    
    def save_model(self):
        """
        存储模型，当存储路径的模型数超过阈值时，删除过期的模型。
        """
        if self.accelerator.is_main_process:
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)

            while len(os.listdir(self.work_dir)) >= self.config['train'].get('save_total_limit', 100):
                all_checkpoints_files = sorted(                                                 # 按照文件修改时间排序
                    os.listdir(self.work_dir),  
                    key=lambda x: os.path.getmtime(os.path.join(self.work_dir, x))
                )
                outdated_model = os.path.join(self.work_dir, all_checkpoints_files[0])
                shutil.rmtree(outdated_model)
                self.accelerator.print(f'[-] Deleted Outdated Model: {outdated_model}.')
        
        start = time.time()
        deep_speed_stage = os.environ.get('ACCELERATE_DEEPSPEED_ZERO_STAGE', '')
        stage3_save_16bit_model = os.environ.get('ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL', 'false')

        if stage3_save_16bit_model in ['true', 'True']:
            stage3_save_16bit_model = True
        elif stage3_save_16bit_model in ['false', 'False']:
            stage3_save_16bit_model = False
        else:
            raise ValueError(f'`stage3_save_16bit_model` in accelerate config should be a bool str, while reveived: {stage3_save_16bit_model}.')
        
        if deep_speed_stage == '3':                                        # Stage3 需要走单独存储的方式
            if stage3_save_16bit_model:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                cur_save_dir = os.path.join(self.work_dir, f'checkpoint-{self.data_step}')
                unwrapped_model.save_pretrained(
                    cur_save_dir, 
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    state_dict=self.accelerator.get_state_dict(self.model)
                )
                self.accelerator.print(f'[+] Stage {deep_speed_stage}: Checkpoint has saved at {cur_save_dir}.')
            else:
                self.model.save_checkpoint(self.work_dir, f'checkpoint-{self.data_step}')
                self.accelerator.print(f'[+] Stage 3: Checkpoint has saved at {os.path.join(self.work_dir, f"checkpoint-{self.data_step}")}.')
            self.accelerator.print(f'Save ckpt used: {round(time.time() - start, 2)}s.')
        else:
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                cur_save_dir = os.path.join(self.work_dir, f'checkpoint-{self.data_step}')
                unwrapped_model.save_pretrained(
                    cur_save_dir, 
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    state_dict=self.accelerator.get_state_dict(self.model)
                )
                self.accelerator.print(f'[+] Stage {deep_speed_stage}: Checkpoint has saved at {cur_save_dir}.')
                self.accelerator.print(f'Save ckpt used: {round(time.time() - start, 2)}s.')

    def train_step(self, batch):
        """
        单步更新。
        """
        out = self.model(**batch)
        total_loss = out.loss
        total_loss = total_loss.mean()                                  # mean() to average on multi-gpu parallel training
        losses = {"total_loss": total_loss}
        self.accelerator.backward(total_loss)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()
        return losses

    def train(self):
        """
        训练loop。
        """
        self.get_optimizer()
        self.get_lr_scheduler()
        self.prepare()
        self.start_time = time.time()
        self.epoch = 0
        self.data_step = 0

        loss_dict_list = {}
        with tqdm(total=self.config["train"]["num_training_steps"]) as pbar:
            while True:
                if self.data_step >= self.config["train"]["num_training_steps"]:
                    break

                if self.epoch == 0:
                    train_loader = self.train_loader_skipped
                else:
                    train_loader = self.train_loader
                
                for batch in train_loader:
                    if self.data_step >= self.config["train"]["num_training_steps"]:
                        break

                    if (
                        self.data_step % self.eval_interval == 0
                        and self.train_and_eval
                    ):
                        self.accelerator.wait_for_everyone()           # stage 3 needs all model params on all devices
                        self.eval()
                    
                    if (
                        self.data_step % self.save_interval == 0
                    ):
                        self.save_model()

                    for k, v in batch.items():
                        batch[k] = v.to(
                            self.accelerator.device, 
                            non_blocking=True
                        )

                    self.model.train()
                    with self.accelerator.accumulate(self.model):
                        losses = self.train_step(batch)
                        
                        if self.accelerator.is_main_process:
                            for k, v in losses.items():
                                if k not in loss_dict_list:
                                    loss_dict_list[k] = []
                                loss_dict_list[k].append(v.cpu().item())
                        
                        if self.accelerator.sync_gradients:
                            self.global_step += 1
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), 
                                1.0
                            )
                    
                    if (
                        self.data_step % self.log_interval == 0
                        and self.accelerator.is_main_process
                    ):
                        self.local_log(loss_dict_list)
                        loss_dict_list = {}
                
                    self.data_step += 1
                    if self.accelerator.is_main_process:
                        pbar.update(1)

                self.epoch += 1
                self.accelerator.print(f'Start {self.epoch - 1} -> {self.epoch} epoch...')  

    def local_log(
        self, 
        loss_dict_list
    ):
        """
        log to local file.

        Args:
            loss_dict_list (_type_): {
                'total_loss': [7.878, 6.79, 6.92, ...]
            }
        """
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        tokens = (
            self.config["train"]["train_batch_size"]
            * self.log_interval
            * self.config["data"]["seq_length"]
        )
        
        token_per_second = 0 if self.data_step == 0 else tokens / cost_time
        self.writer.add_scalar(
            f"Training/Token per second per gpu",
            token_per_second,
            self.data_step
        )
        self.accelerator.print(f'token per second per gpu: {token_per_second}')
        
        for k, values in loss_dict_list.items():
            self.writer.add_scalar(
                "Training/{}".format(k),
                sum(values) / len(values),
                self.data_step
            )
            self.accelerator.print(f'{k}: {sum(values) / len(values)}')
        
        current_lr = self.optim.param_groups[0]["lr"]
        
        self.writer.add_scalar(
            "Training/LR",
            current_lr,
            self.data_step
        )
        self.accelerator.print(f"lr: {current_lr}")

        current_tokens = (
            self.config["train"]["train_batch_size"]
            * self.data_step
            * self.config["data"]["seq_length"]
        )
        current_tokens = current_tokens / 1e9               # 统计已训练的token数，转换为B单位
        
        self.writer.add_scalar(
            "Training/Trained tokens per gpu(B)",
            current_tokens,
            self.data_step
        )
        self.accelerator.print(f"trained tokens per gpu(B): {current_tokens}")

        if self.optim.scaler is not None:
            self.writer.add_scalar(
                "Training/Loss Scale",
                self.optim.scaler.get_scale(),
                self.data_step
            )
            self.accelerator.print(f"loss scale: {self.optim.scaler.get_scale()}")
        
        self.writer.record()
    
    def eval(self):
        """
        评测函数。
        """
        self.model.eval()
        eval_methods = self.config['eval'].get('eval_methods', [])
        if not eval_methods:
            raise ValueError(f"You must specify eval_methods in {self.all_eval_methods}, otherwise you could set `train_and_eval` to False.")
    
        print(f'Start to eval_methods: {eval_methods}...')
        for method in eval_methods:
            if method not in self.all_eval_methods:
                raise ValueError(f'No valid eval_method called {method}, which should in {self.all_eval_methods}.')
            
            if method == 'nll':
                nll_loss_dict = eval_nll(
                    self.model, 
                    self.tokenizer, 
                    self.accelerator.device, 
                    self.config['eval'].get('nll_dataset', {}), 
                    self.config["data"]["seq_length"],
                )

                if self.accelerator.is_main_process:
                    for k, v in nll_loss_dict.items():
                        self.writer.add_scalar(
                            "eval/NLL/{}".format(k),
                            v,
                            self.data_step
                        )                
                    self.accelerator.print(f'NLL: {nll_loss_dict}')
                    self.writer.record()

            elif method == 'generation_eval':
                res_list = eval_generation(
                    self.model, 
                    self.tokenizer, 
                    self.accelerator.device, 
                    self.config['eval'].get('generation_dataset', {})
                )

                if self.accelerator.is_main_process:
                    eval_save_path = self.config['eval']['genration_eval_save_path']
                    
                    if not os.path.exists(eval_save_path):
                        os.makedirs(eval_save_path)

                    cur_save_file = os.path.join(eval_save_path, f'step_{self.data_step}.json')
                    json.dump(res_list, open(cur_save_file, 'w', encoding='utf8'), ensure_ascii=False)
                    self.accelerator.print(f'[+] Eval Performance has been saved at {cur_save_file}.')
            
            elif method == 'single_choice_eval':
                metrics_dict = eval_single_choice(
                    self.model,
                    self.tokenizer,
                    self.accelerator.device, 
                    self.config['eval'].get('single_choice_dataset', {}),
                )

                if self.accelerator.is_main_process:
                    acc_dict = dict([(k, v['acc']) for k, v in metrics_dict.items()])
                    for k, v in acc_dict.items():
                        self.writer.add_scalar(
                            "eval/SC-Acc/{}".format(k),
                            v,
                            self.data_step
                        )

                    self.accelerator.print(f'SingleChoiceEval: {acc_dict}')
                    self.writer.record()
        
        self.model.train()