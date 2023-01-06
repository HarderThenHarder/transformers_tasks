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

BLEU指标。

Reference:
    https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/metrics/bleu.py

Author: pankeyu
Date: 2022/1/5
"""
import sys
import math
from typing import List
from collections import defaultdict

import numpy as np


def default_trans_func(output, label, seq_mask, vocab):
    seq_mask = np.expand_dims(seq_mask, axis=2).repeat(output.shape[2], axis=2)
    output = output * seq_mask
    idx = np.argmax(output, axis=2)
    prediction, references = [], []
    for i in range(idx.shape[0]):
        token_list = []
        for j in range(idx.shape[1]):
            if seq_mask[i][j][0] == 0:
                break
            token_list.append(vocab[idx[i][j]])
        prediction.append(token_list)

    label = np.squeeze(label, axis=2)
    for i in range(label.shape[0]):
        token_list = []
        for j in range(label.shape[1]):
            if seq_mask[i][j][0] == 0:
                break
            token_list.append(vocab[label[i][j]])

        references.append([token_list])
    return prediction, references


def get_match_size(prediction_ngram, refs_ngram):
    ref_set = defaultdict(int)
    for ref_ngram in refs_ngram:
        tmp_ref_set = defaultdict(int)
        for ngram in ref_ngram:
            tmp_ref_set[tuple(ngram)] += 1
        for ngram, count in tmp_ref_set.items():
            ref_set[tuple(ngram)] = max(ref_set[tuple(ngram)], count)
    prediction_set = defaultdict(int)
    for ngram in prediction_ngram:
        prediction_set[tuple(ngram)] += 1
    match_size = 0
    for ngram, count in prediction_set.items():
        match_size += min(count, ref_set.get(tuple(ngram), 0))
    prediction_size = len(prediction_ngram)
    return match_size, prediction_size


def get_ngram(sent, n_size, label=None):
    def _ngram(sent, n_size):
        ngram_list = []
        for left in range(len(sent) - n_size):
            ngram_list.append(sent[left : left + n_size + 1])
        return ngram_list

    ngram_list = _ngram(sent, n_size)
    if label is not None:
        ngram_list = [ngram + "_" + label for ngram in ngram_list]
    return ngram_list


class BLEU(object):
    """
    BLEU 评估器。

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/metrics/bleu.py

    Examples:
        from bleu_metrics import BLEU

        bleu = BLEU(n_size=1)
        prediction = ["猫", "在", "桌", "上"]
        references = [["猫", "在", "树", "上"]]
        bleu.add_inst(prediction, references)
        print(bleu.score())                     # 0.75
    """

    def __init__(self, trans_func=None, vocab=None, n_size=4, name="bleu"):
        """
        Args:
            trans_func (callable, optional): `trans_func` transforms the network
                output to string to calculate.
            vocab (dict|paddlenlp.data.vocab, optional): Vocab for target language.
                If `trans_func` is None and BLEU is used as `paddle.metric.Metric`
                instance, `default_trans_func` will be performed and `vocab` must
                be provided.
            n_size (int, optional): Number of gram for BLEU metric. Defaults to 4.
            weights (list, optional): The weights of precision of each gram.
                Defaults to None.
            name (str, optional): Name of `paddle.metric.Metric` instance.
                Defaults to "bleu".
        """
        super().__init__()
        weights = [1 / n_size for _ in range(n_size)]
        self._name = name
        self.match_ngram = {}
        self.prediction_ngram = {}
        self.weights = weights
        self.bp_r = 0
        self.bp_c = 0
        self.n_size = n_size
        self.vocab = vocab
        self.trans_func = trans_func

    def update(self, output, label, seq_mask=None):
        if self.trans_func is None:
            if self.vocab is None:
                raise AttributeError(
                    "The `update` method requires users to provide `trans_func` or `vocab` when initializing BLEU."
                )
            prediction_list, references = default_trans_func(output, label, seq_mask=seq_mask, vocab=self.vocab)
        else:
            prediction_list, references = self.trans_func(output, label, seq_mask)
        if len(prediction_list) != len(references):
            raise ValueError("Length error! Please check the output of network.")
        for i in range(len(prediction_list)):
            self.add_inst(prediction_list[i], references[i])

    def add_instance(self, prediction: List[str], references: List[List[str]]):
        """
        Update the states based on a pair of prediction and references.

        Args:
            prediction (list): Tokenized prediction sentence.
            references (list of list): List of tokenized ground truth sentences.
        """
        for n_size in range(self.n_size):
            self.count_ngram(prediction, references, n_size)
        self.count_bp(prediction, references)

    def count_ngram(self, prediction, references, n_size):
        prediction_ngram = get_ngram(prediction, n_size)
        refs_ngram = []
        for ref in references:
            refs_ngram.append(get_ngram(ref, n_size))
        if n_size not in self.match_ngram:
            self.match_ngram[n_size] = 0
            self.prediction_ngram[n_size] = 0
        match_size, prediction_size = get_match_size(prediction_ngram, refs_ngram)

        self.match_ngram[n_size] += match_size
        self.prediction_ngram[n_size] += prediction_size

    def count_bp(self, prediction, references):
        self.bp_c += len(prediction)
        self.bp_r += min([(abs(len(prediction) - len(ref)), len(ref)) for ref in references])[1]

    def reset(self):
        self.match_ngram = {}
        self.prediction_ngram = {}
        self.bp_r = 0
        self.bp_c = 0

    def accumulate(self):
        """
        Calculates and returns the final bleu metric.

        Returns:
            Tensor: Returns the accumulated metric `bleu` and its data type is float64.
        """
        prob_list = []
        for n_size in range(self.n_size):
            try:
                if self.prediction_ngram[n_size] == 0:
                    _score = 0.0
                else:
                    _score = self.match_ngram[n_size] / float(self.prediction_ngram[n_size])
            except:
                _score = 0
            if _score == 0:
                _score = sys.float_info.min
            prob_list.append(_score)

        logs = math.fsum(w_i * math.log(p_i) for w_i, p_i in zip(self.weights, prob_list))
        bp = math.exp(min(1 - self.bp_r / float(self.bp_c), 0))
        bleu = bp * math.exp(logs)
        return bleu

    def compute(self):
        return self.accumulate()

    def name(self):
        return self._name


if __name__ == '__main__':
    blue = BLEU(n_size=1)
    prediction = list("猫坐在椅子上")
    references = [list("猫坐在树上")]
    blue.add_instance(prediction=prediction, references=references)
    print(blue.compute())
