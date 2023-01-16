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

Reinforcemnet Learning from Human Feedback 终端版。

Author: pankeyu
Date: 2022/12/27
"""
import os
import time
import random
from rich import box
from rich.table import Table
from rich.align import Align
from rich.console import Console

import torch
import torch.nn.functional as F
from trl.ppo import PPOTrainer
from trl.gpt2 import GPT2HeadWithValueModel
from transformers import AutoTokenizer
from transformers import top_k_top_p_filtering

from iTrainingLogger import iSummaryWriter


MODEL_CONFIG = {
    'model_name': 'uer/gpt2-chinese-cluecorpussmall',
    'device': 'cuda:0'
}
MIN_REWARD = -2.0
MAX_REWARD = 2.0

LOG_PATH = './logs'
LOG_NAME = 'Terminal-Human-Feedback'
writer = iSummaryWriter(log_path=LOG_PATH, log_name=LOG_NAME)

prompts = [
            '刚收到货，感觉',
            '这部电影很',
            '说实话，真的很',
            '这次购物总的来说体验很'
        ]


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """
    根据prompt生成答案。

    Args:
        model (_type_): _description_
        queries (_type_): _description_
        txt_len (int, optional): _description_. Defaults to 20.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    input_ids = queries
    device = MODEL_CONFIG['device']
    for _ in range(txt_len):
        outputs = model(input_ids.to(device))
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids.to(device), next_token.unsqueeze(-1)], dim=-1).cpu()
    return input_ids[:, -txt_len:]


def main():
    """
    主函数。
    """
    console = Console()
    table = Table(show_footer=False)
    table.width = console.width
    table.box = box.SQUARE
    table.row_styles = ["none", "dim"]
    console.clear()

    # add title
    table.title = (
            "[bold not italic]:robot:[/] Reinforcemnet Learning from Human Feedback - Terminal"
        )

    # add column (first line)
    table.add_column("config/key", no_wrap=True)
    table.add_column("config/value", no_wrap=True)
    
    # add config row to table
    for k, v in MODEL_CONFIG.items():
        table.add_row(k, v)
    table.add_row('log path', os.path.join(LOG_PATH, LOG_NAME))
    table.add_row('min ~ max reward', f'{MIN_REWARD} ~ {MAX_REWARD}')
    table.add_row('prompts', f'{prompts}')
    table.caption = "You can change config in [b not dim]Source Code[/]"
    
    table.columns[0].style = "bright_red"
    table.columns[0].header_style = "bold bright_red"
    table.columns[1].style = "bright_green"
    table.columns[1].header_style = "bold bright_green"
    table_centered = Align.center(table)
    console.print(table_centered)

    with console.status("[bold bright_green]Initializing Model & Env..."):
        model = GPT2HeadWithValueModel.from_pretrained(MODEL_CONFIG['model_name']).to(MODEL_CONFIG['device'])
        ref_model = GPT2HeadWithValueModel.from_pretrained(MODEL_CONFIG['model_name']).to(MODEL_CONFIG['device'])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
        ppo_trainer = PPOTrainer(
            model,
            ref_model,
            tokenizer,
            **ppo_config
        )
        console.log('[bold magenta][Done] Initialized Model & Env.')

    step = 1
    t = time.time()
    while True:
        current_prompt = random.choice(prompts)
        console.print(f'[Step {step}]')
        console.print(f'[bright_yellow]prompt>>> {current_prompt}[/bright_yellow]')
        console.print('generating results...', end='\r')
        query_tensor = tokenizer.encode(current_prompt, return_tensors="pt").to(MODEL_CONFIG['device'])
        response_tensor = respond_to_batch(model, query_tensor)
        response_txt = tokenizer.decode(response_tensor[0, :].to('cpu'))

        console.print(f'[bright_blue]result>>> {response_txt}[/bright_blue]')
        reward_txt = input(f'Reward ({MIN_REWARD} ~ {MAX_REWARD}): ')
        while True:
            try:
                reward_f = float(reward_txt)
                if MIN_REWARD <= reward_f <= MAX_REWARD:
                    break
                else:
                    reward_txt = input(f'Reward ({MIN_REWARD} ~ {MAX_REWARD}): ')
            except:
                reward_txt = input(f'Reward ({MIN_REWARD} ~ {MAX_REWARD}): ')
        reward = [torch.tensor(reward_f).to(MODEL_CONFIG['device'])]

        with console.status("[bold bright_green]Updating Model..."):
            ppo_trainer.step([query_tensor[0]], [response_tensor[0].to(MODEL_CONFIG['device'])], reward)
        
        writer.add_scalar('reward history', reward_f, step)
        writer.add_scalar('label time used', time.time() - t, step)
        writer.record()
        t = time.time()
        step += 1


if __name__ == '__main__':
    main()
