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

数据集生成的相关工具。

Author: pankeyu
Date: 2022/10/26
"""
import os
import random

from tqdm import tqdm
from rich import print


def split_dataset(
    origin_file: str, output_path: str, split_ratios=[0.7, 0.2, 0.1], shuffle=True
):
    """
    将一整份数据集切成train, dev, test三份。

    Args:
        origin_file (str): 全量数据集文件 -> e.g. total_dataset.txt
        output_path (str): 切分后的数据集存放地址 -> e.g. data/
        split_ratios (list, optional): train/dev/test 比例. Defaults to [0.7, 0.2, 0.1].
    """
    assert sum(split_ratios) - 1 < 1e-5, f"分割比例之和必须等于1，当前输入和为 {sum(split_ratios)}。"

    lines = open(origin_file, "r", encoding="utf8").readlines()
    total_samples = len(lines)
    if shuffle:
        random.shuffle(lines)

    train_num = int(total_samples * split_ratios[0])
    dev_num = int(total_samples * split_ratios[1])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    suffix = origin_file.split(".")[-1]
    with open(
        os.path.join(output_path, f"train.{suffix}"), "w", encoding="utf8"
    ) as trf, open(
        os.path.join(output_path, f"dev.{suffix}"), "w", encoding="utf8"
    ) as devf, open(
        os.path.join(output_path, f"test.{suffix}"), "w", encoding="utf8"
    ) as tef:
        i = 0
        for line in tqdm(lines, colour="green"):
            if i < train_num:
                trf.write(line)
            elif i < train_num + dev_num:
                devf.write(line)
            else:
                tef.write(line)
            i += 1

    print("=" * 50 + " Done " + "=" * 50)
    print(
        {
            "train samples": train_num,
            "dev samples": dev_num,
            "test samples": total_samples - train_num - dev_num,
        }
    )
    print(f"datasets have been saved at: {output_path}.")


if __name__ == "__main__":
    split_dataset(
        "total_dataset.txt",
        "data/",
        split_ratios=[0.8, 0.199, 0.001],
        shuffle=True,
    )
