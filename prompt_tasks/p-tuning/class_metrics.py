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

（多）分类问题下的指标评估（acc, precision, recall, f1）。

Author: pankeyu
Date: 2022/11/20
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


class ClassEvaluator(object):

    def __init__(self):
        """
        init func.
        """
        self.goldens = []
        self.predictions = []
    
    def add_batch(self, pred_batch: List[List], gold_batch: List[List]):
        """
        添加一个batch中的prediction和gold列表，用于后续统一计算。

        Args:
            pred_batch (list): 模型预测标签列表, e.g. -> [0, 0, 1, 2, 0, ...] or [['体', '育'], ['财', '经'], ...]
            gold_batch (list): 真实标签标签列表, e.g. -> [1, 0, 1, 2, 0, ...] or [['体', '育'], ['财', '经'], ...]
        """
        assert len(pred_batch) == len(gold_batch), \
            f"@params pred_spans_batch(len: {len(pred_batch)}) does not match @param gold_spans_batch(len: {len(gold_batch)})"

        if type(gold_batch[0]) in [list, tuple]:                                    # 若遇到多个子标签构成一个标签的情况
            pred_batch = [','.join([str(e) for e in ele]) for ele in pred_batch]     # 将所有的label拼接为一个整label: ['体', '育'] -> '体育'
            gold_batch = [','.join([str(e) for e in ele]) for ele in gold_batch]
        self.goldens.extend(gold_batch)
        self.predictions.extend(pred_batch)

    def compute(self, round_num=2) -> dict:
        """
        根据当前类中累积的变量值，计算当前的P, R, F1。
        
        Args:
            round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。

        Returns:
            dict -> {
                'accuracy': 准确率,
                'precision': 精准率,
                'recall': 召回率,
                'f1': f1值,
                'class_metrics': {
                    '0': {
                            'precision': 该类别下的precision, 
                            'recall': 该类别下的recall,
                            'f1': 该类别下的f1
                        },
                    ...
                }
            }
        """
        classes, class_metrics, res = sorted(list(set(self.goldens) | set(self.predictions))), {}, {}
        res['accuracy'] = round(accuracy_score(self.goldens, self.predictions), round_num)      # 构建全局指标
        res['precision'] = round(precision_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['recall'] = round(recall_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['f1'] = round(f1_score(self.goldens, self.predictions, average='weighted'), round_num)

        try:
            conf_matrix = np.array(confusion_matrix(self.goldens, self.predictions))                # (n_class, n_class)
            assert conf_matrix.shape[0] == len(classes), f"confusion_matrix shape ({conf_matrix.shape[0]}) doesn't match labels number ({len(classes)})!"
            for i in range(conf_matrix.shape[0]):                                                   # 构建每个class的指标
                precision = 0 if sum(conf_matrix[:, i]) == 0 else conf_matrix[i, i] / sum(conf_matrix[:, i])
                recall = 0 if sum(conf_matrix[i, :]) == 0 else conf_matrix[i, i] / sum(conf_matrix[i, :])
                f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                class_metrics[classes[i]] = {
                    'precision': round(precision, round_num),
                    'recall': round(recall, round_num),
                    'f1': round(f1, round_num)
                }
            res['class_metrics'] = class_metrics
        except Exception as e:
            print(f'[Warning] Something wrong when calculate class_metrics: {e}')
            print(f'-> goldens: {set(self.goldens)}')
            print(f'-> predictions: {set(self.predictions)}')
            print(f'-> diff elements: {set(self.predictions) - set(self.goldens)}')
            res['class_metrics'] = {}
        
        return res

    def reset(self):
        """
        重置积累的数值。
        """
        self.goldens = []
        self.predictions = []


if __name__ == '__main__':
    from rich import print

    metric = ClassEvaluator()
    metric.add_batch(
        [['财', '经'], ['财', '经'], ['体', '育'], ['体', '育'], ['计', '算', '机']],
        [['体', '育'], ['财', '经'], ['体', '育'], ['计', '算', '机'], ['计', '算', '机']],
    )
    # metric.add_batch(
    #     [0, 0, 1, 1, 0],
    #     [1, 1, 1, 0, 0]
    # )
    print(metric.compute())