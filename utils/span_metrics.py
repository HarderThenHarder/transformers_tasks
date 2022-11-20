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

指标计算函数集合，包括Span抽取的Precision, Recall, F1的计算。

Author: pankeyu
Date: 2022/11/04
"""
from typing import List


class SpanEvaluator(object):
    """
    用于计算模型Span提取的准确率。

    Code Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/metrics/span.py
    """

    def __init__(self):
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def add_batch(self, pred_spans_batch: List[List[str]], gold_spans_batch: List[List[str]]):
        """
        输入预测的span列表和真实标签span列表，计算正确的数并更新成员变量中的值。

        Args:
            pred_spans (List[List[str]]): 模型预测span列表, e.g. -> [
                                                                    ['电视剧', '电影'], 
                                                                    ['蛋糕', '面包'],
                                                                    ...
                                                                ]
            gold_spans (List[List[str]]): 真实标签span列表, e.g. -> [
                                                                    ['电视剧'], 
                                                                    ['蛋糕', '面包', '青团'],
                                                                    ...
                                                                ]
        """
        assert len(pred_spans_batch) == len(gold_spans_batch), \
            f"@params pred_spans_batch(len: {len(pred_spans_batch)}) does not match @param gold_spans_batch(len: {len(gold_spans_batch)})"

        for pred_spans, gold_spans in zip(pred_spans_batch, gold_spans_batch):
            pred_set = set(pred_spans)                              # 得到模型输出的span集合(set)
            label_set = set(gold_spans)                             # 得到标签中正确的span集合(set)
            self.num_correct_spans += len(pred_set & label_set)     # 计算正确预测的span集合(两个集合求交集)
            self.num_infer_spans += len(pred_set)
            self.num_label_spans += len(label_set)

    def compute(self, round_num=2) -> dict:
        """
        根据当前类中累积的变量值，计算当前的P, R, F1。
        
        Args:
            round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。

        Returns:
            dict -> {
                'precision': 精准率,
                'recall': 召回率,
                'f1': f1值
            }
        """
        precision = float(self.num_correct_spans / self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans / self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall / (precision + recall)) if self.num_correct_spans else 0.
        return {
            'precision': round(precision, round_num), 
            'recall': round(recall, round_num), 
            'f1': round(f1_score, round_num)
        }

    def reset(self):
        """
        重置积累的数值。
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0


class MultiTypesSpanEvaluator(object):
    """
    当存在多个类别时，分类别统计P、R、F1。

    Args:
        object (_type_): _description_
    """

    def __init__(self):
        """
        init func.
        """
        self.type_metric = {}

    def add_batch(self, types: List[str], pred_spans_batch: List[List[str]], gold_spans_batch: List[List[str]]):
        """
        批量添加多个type的span预测/真实span列表，更新成员变量的值。

        Args:
            types (List[str]): ['作品', '食物', ...]
            pred_spans (List[List[str]]): 模型预测span列表, e.g. -> [
                                                                    ['电视剧', '电影'], 
                                                                    ['蛋糕', '面包'],
                                                                    ...
                                                                ]
            gold_spans (List[List[str]]): 真实标签span列表, e.g. -> [
                                                                    ['电视剧'], 
                                                                    ['蛋糕', '面包', '青团'],
                                                                    ...
                                                                ]
        """
        assert len(types) == len(pred_spans_batch) == len(gold_spans_batch), \
            f"@params types(len: {len(types)}) does not match @param pred_spans_batch(len: {len(pred_spans_batch)}), @param gold_spans_batch(len: {len(gold_spans_batch)})"
        
        for _type, pred_spans, gold_spans in zip(types, pred_spans_batch, gold_spans_batch):
            if _type not in self.type_metric:
                self.type_metric[_type] = SpanEvaluator()           # 当出现一个新的type类型时，往字典中新建一个该type的span统计器
            self.type_metric[_type].add_batch([pred_spans], [gold_spans])
    
    def compute(self, round_num=2) -> dict:
        """
        根据当前类中累积的变量值，计算当前的P, R, F1。
        
        Args:
            round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。

        Returns:
            dict -> {
                    'type1': 
                        {
                            'precision': 精准率,
                            'recall': 召回率,
                            'f1': f1值
                        },
                    'type2': 
                        {
                            'precision': 精准率,
                            'recall': 召回率,
                            'f1': f1值
                        },
                    ...
            }
        """
        res = {}
        for _type in self.type_metric:
            res[_type] = self.type_metric[_type].compute(round_num=round_num)
        return res

    def reset(self):
        """
        重置积累的数值。
        """
        self.type_metric.clear()


if __name__ == '__main__':
    from rich import print

    metric = SpanEvaluator()
    metric.add_batch(
        [['电视剧', '电影'], ['蛋糕', '面包']],
        [['电视剧'], ['蛋糕', '面包', '青团', '馒头']]
    )
    print(metric.compute())

    multi_class_metric = MultiTypesSpanEvaluator()
    multi_class_metric.add_batch(
        ['subject', 'types'],
        [['电视剧', '电影'], ['蛋糕', '面包']],
        [['电视剧'], ['蛋糕', '面包', '青团', '馒头']]
    )
    print(multi_class_metric.compute())
