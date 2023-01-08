<div align=center> 

<img src="assets/icon.png">

[![Author](https://img.shields.io/badge/Author-Pankeyu-green.svg "Author")](https://www.zhihu.com/column/c_1451236880973426688) [![OS](https://img.shields.io/badge/OS-Linux/Windows/Mac-red.svg "OS")](./) [![Based](https://img.shields.io/badge/Based-huggingface_transformers-blue.svg "OS")](./)

</div>

基于 transformers 库实现的多种 NLP 任务。

<br>

💡 已实现的任务模型：

#### 1. 文本匹配（Text Matching）

| 模型  | 传送门  |
|---|---|
| 概览  | [[这里]](./text_matching/readme.md) |
| PointWise（单塔）  | [[这里]](./text_matching/train_pointwise.sh) |
| DSSM（双塔）  | [[这里]](./text_matching/train_dssm.sh) |
| Sentence Bert（双塔）  | [[这里]](./text_matching/train_sentence_transformer.sh) |


#### 2. 信息抽取（Information Extraction）

| 模型  | 传送门  |
|---|---|
| UIE  | [[这里]](./UIE/readme.md) |


#### 3. Prompt任务（Prompt Tasks）

| 模型  | 传送门  |
|---|---|
| PET  | [[这里]](./prompt_tasks/PET/readme.md) |
| p-tuning  | [[这里]](./prompt_tasks/p-tuning/readme.md) |


#### 4. 文本分类（Text Classification）

| 模型  | 传送门  |
|---|---|
| BERT-CLS  | [[这里]](./text_classification/train.sh) |


#### 5. 强化学习 & 语言模型（Reinforcement Learning & Language Model）

| 模型  | 传送门  |
|---|---|
| PPO-GPT2  | [[这里]](./RLHF/readme.md) |

#### 6. 文本生成（Text Generation）

| 模型  | 传送门  |
|---|---|
| 问答模型 | [[这里]](./answer_generation/readme.md) |