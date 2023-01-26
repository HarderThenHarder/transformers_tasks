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

随机 MASK 文本中的一段，生成 filling 模型训练数据集。

Author: pankeyu
Date: 2023/01/26
"""
import random

import jieba
from tqdm import tqdm
from rich import print


def generate_mask_fill_dataset():
    """
    在原始文本中随机生成[MASK]token。
    """
    MIN_MASK_LEN_RATIO = 0.1                        # 随机MASK SPAN的最小长度（词粒度），最短n个词
    MAX_MASK_LEN_RATIO = 0.5                        # 随机MASK SPAN的最大长度（词粒度），最长N个词
    RANDOM_MASK_PER_SAMPLE = 2                      # 每个句子随机MASK几次
    
    samples = []
    with open('data/dataset_text.txt', 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = jieba.lcut(line.strip())
            # print('line: ', line)
            MIN_MASK_LEN = int(len(line) * MIN_MASK_LEN_RATIO)
            MAX_MASK_LEN = int(len(line) * MAX_MASK_LEN_RATIO)
            if len(line) <= MAX_MASK_LEN:
                continue
            for _ in range(RANDOM_MASK_PER_SAMPLE):
                random_mask_span_length = random.randint(MIN_MASK_LEN, MAX_MASK_LEN)
                # print('random_mask_span_length: ', random_mask_span_length)
                random_mask_span_start_index = random.randint(0, len(line) - random_mask_span_length)
                masked_text = line[:random_mask_span_start_index] + ['[MASK]'] + line[random_mask_span_start_index + random_mask_span_length:]
                masked_label = line[random_mask_span_start_index:random_mask_span_start_index + random_mask_span_length]
                masked_text = ''.join(masked_text)
                masked_text = f'"{masked_text}"中[MASK]位置的文本是：'
                masked_label = ''.join(masked_label)
                sample = f'{masked_text}\t{masked_label}\n'
                samples.append(sample)
                # print(sample)
                # exit()

    print('Samples Len: ', len(samples))
    print(samples[:10])
    train_file, dev_file = 'data/train.tsv', 'data/dev.tsv'
    train_test_split_ratio = 0.9
    train_sample_count = int(train_test_split_ratio * len(samples))
    random.shuffle(samples)
    with open(train_file, 'w', encoding='utf8') as f:
        for sample in samples[:train_sample_count]:
            f.write(f'{sample}')
    with open(dev_file, 'w', encoding='utf8') as f:
        for sample in samples[train_sample_count:]:
            f.write(f'{sample}')
    
    print(f'[Done] File has saved at {train_file} {dev_file}.')


if __name__ == '__main__':
    generate_mask_fill_dataset()
