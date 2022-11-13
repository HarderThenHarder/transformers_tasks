### P-tuning：Auto Learning prompt pattern
---

P-tuning 是 prompt learning 下的一个重要分支，关于 P-tuning 的详细介绍在：[这里](https://zhuanlan.zhihu.com/p/583022692)。


### 1. 环境安装
本项目基于 `pytorch` + `transformers` 实现，运行前请安装相关依赖包：

```sh
pip install -r ../../requirements.txt
```

### 2. 数据集准备
项目中提供了一部分示例数据，根据用户评论预测用户评论的物品类别（分类任务），数据在 `data/comment_classify` 。

若想使用`自定义数据`训练，只需要仿照示例数据构建数据集即可：

```
...
水果	什么苹果啊，都没有苹果味，怪怪的味道，而且一点都不甜，超级难吃！
...
```

每一行用 `\t` 分隔符分开，前半部分为`标签（label）`，后半部分为`原始输入`。

> Note: 数据中所有的标签必须拥有「相同长度」！不能出现标签长度不同的情况：例如 -> '计算机'、'水果'...，遇到标签长度不等情况时需要先将标签数据处理为等长再训练。


### 3. 模型训练
修改训练脚本 `train.sh` 里的对应参数, 开启模型训练：

```sh
python p_tuning.py \
    --model "bert-base-chinese" \   # backbone
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/" \
    --batch_size 16 \
    --max_seq_len 128 \
    --valid_steps 20  \
    --logging_steps 5 \
    --num_train_epochs 50 \
    --device "cuda:0"              # 指定使用哪块gpu
```
正确开启训练后，终端会打印以下信息：

```sh
...
global step 5, epoch: 1, loss: 6.50529, speed: 4.25 step/s
global step 10, epoch: 2, loss: 4.77712, speed: 6.36 step/s
global step 15, epoch: 3, loss: 3.55371, speed: 6.19 step/s
global step 20, epoch: 4, loss: 2.71686, speed: 6.38 step/s
Evaluation precision: 0.70000, recall: 0.69000, F1: 0.69000
best F1 performence has been updated: 0.00000 --> 0.69000
global step 25, epoch: 6, loss: 2.20488, speed: 6.21 step/s
global step 30, epoch: 7, loss: 1.84836, speed: 6.22 step/s
global step 35, epoch: 8, loss: 1.58520, speed: 6.22 step/s
global step 40, epoch: 9, loss: 1.38746, speed: 6.27 step/s
Evaluation precision: 0.75000, recall: 0.75000, F1: 0.75000
best F1 performence has been updated: 0.69000 --> 0.75000
global step 45, epoch: 11, loss: 1.23437, speed: 6.14 step/s
global step 50, epoch: 12, loss: 1.11103, speed: 6.16 step/s
...
```

在 `logs/sentiment_classification` 文件下将会保存训练曲线图：

<img src='assets/train_log.png'></img>


### 4. 模型预测

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```python
...
contents = ["苹果卖相很好，而且很甜，很喜欢这个苹果，下次还会支持的", "这破笔记本速度太慢了，卡的不要不要的"]   # 自定义评论
res = inference(contents)       # 推测评论类型
...
```

运行推理程序：

```sh
python inference.py
```

得到以下推理结果：

```sh
inference label(s): ['水果', '电脑']
```
