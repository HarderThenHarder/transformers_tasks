## 文本匹配任务（Text Matching）

---

文本匹配多用于计算两个文本之间的相似度，关于文本匹配的详细介绍在：[这里]()。

本项目对3种常用的文本匹配的方法进行实现：PointWise（单塔）、DSSM（双塔）、Sentence BERT（双塔）。

### 1. PointWise（单塔）

#### 1.1 环境安装

本项目基于 `pytorch` + `transformers` 实现，运行前请安装相关依赖包：

```sh
pip install -r ../requirements.txt
```

#### 1.2 数据集准备

项目中提供了一部分示例数据，我们使用「商品评论」和「商品类别」来进行文本匹配任务，数据在 `data/comment_classify` 。

若想使用`自定义数据`训练，只需要仿照示例数据构建数据集即可：：

```python
衣服：指穿在身上遮体御寒并起美化作用的物品。	为什么是开过的洗发水都流出来了、是用过的吗？是这样子包装的吗？	0
衣服：指穿在身上遮体御寒并起美化作用的物品。	开始买回来大很多 后来换了回来又小了 号码区别太不正规 建议各位谨慎	1
...
```

每一行用 `\t` 分隔符分开，第一部分部分为`商品类型（text1）`，中间部分为`商品评论（text2）`，最后一部分为`商品评论和商品类型是否一致（label）`。

#### 1.3 模型训练
修改训练脚本 `train_pointwise.sh` 里的对应参数, 开启模型训练：

```sh
python train_pointwise.py \
    --model "nghuyong/ernie-3.0-base-zh" \  # backbone
    --train_path "data/comment_classify/train.txt" \    # 训练集
    --dev_path "data/comment_classify/dev.txt" \    #验证集
    --save_dir "checkpoints/comment_classify" \ # 训练模型存放地址
    --img_log_dir "logs/comment_classify" \ # loss曲线图保存位置
    --img_log_name "ERNIE-PointWise" \  # loss曲线图保存文件夹
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
global step 10, epoch: 1, loss: 0.77517, speed: 3.43 step/s
global step 20, epoch: 1, loss: 0.67356, speed: 4.15 step/s
global step 30, epoch: 1, loss: 0.53567, speed: 4.15 step/s
global step 40, epoch: 1, loss: 0.47579, speed: 4.15 step/s
global step 50, epoch: 2, loss: 0.43162, speed: 4.41 step/s
Evaluation precision: 0.88571, recall: 0.87736, F1: 0.88152
best F1 performence has been updated: 0.00000 --> 0.88152
global step 60, epoch: 2, loss: 0.40301, speed: 4.08 step/s
global step 70, epoch: 2, loss: 0.37792, speed: 4.03 step/s
global step 80, epoch: 2, loss: 0.35343, speed: 4.04 step/s
global step 90, epoch: 2, loss: 0.33623, speed: 4.23 step/s
global step 100, epoch: 3, loss: 0.31319, speed: 4.01 step/s
Evaluation precision: 0.96970, recall: 0.90566, F1: 0.93659
best F1 performence has been updated: 0.88152 --> 0.93659
...
```

在 `logs/comment_classify` 文件下将会保存训练曲线图：

<img src='assets/train_log.png'></img>

#### 1.4 模型预测

完成模型训练后，运行 `inference_pointwise.py` 以加载训练好的模型并应用：

```python
...
    test_inference(
        '手机：一种可以在较广范围内使用的便携式电话终端。',     # 第一句话
        '味道非常好，京东送货速度也非常快，特别满意。',        # 第二句话
        max_seq_len=128
    )
...
```

运行推理程序：

```sh
python inference_pointwise.py
```

得到以下推理结果：

```sh
tensor([[ 1.8477, -2.0484]], device='cuda:0')   # 两句话不相似(0)的概率更大
```

### 2. DSSM（双塔）

### 3. Sentence BERT（双塔）