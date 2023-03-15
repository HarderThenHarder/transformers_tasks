## 文本匹配任务（Text Matching）


文本匹配多用于计算两个文本之间的相似度，关于文本匹配的详细介绍在：[这里](https://zhuanlan.zhihu.com/p/585533302?)。

本项目对3种常用的文本匹配的方法进行实现：PointWise（单塔）、DSSM（双塔）、Sentence BERT（双塔）。



## 1. 环境安装

本项目基于 `pytorch` + `transformers` 实现，运行前请安装相关依赖包：

```sh
pip install -r ../../requirements.txt
```

## 2. 数据集准备

项目中提供了一部分示例数据，我们使用「商品评论」和「商品类别」来进行文本匹配任务，数据在 `data/comment_classify` 。

若想使用`自定义数据`训练，只需要仿照示例数据构建数据集即可：

```python
衣服：指穿在身上遮体御寒并起美化作用的物品。	为什么是开过的洗发水都流出来了、是用过的吗？是这样子包装的吗？	0
衣服：指穿在身上遮体御寒并起美化作用的物品。	开始买回来大很多 后来换了回来又小了 号码区别太不正规 建议各位谨慎	1
...
```

每一行用 `\t` 分隔符分开，第一部分部分为`商品类型（text1）`，中间部分为`商品评论（text2）`，最后一部分为`商品评论和商品类型是否一致（label）`。


## 3. 模型训练

### 3.1 PointWise（单塔） 

#### 3.1.1 模型训练

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

<img src='assets/pointwise_train_log.png'></img>

#### 3.1.2 模型推理

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

### 3.2 DSSM（双塔）

#### 3.2.1 模型训练

修改训练脚本 `train_dssm.sh` 里的对应参数, 开启模型训练：

```sh
python train_dssm.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/dssm" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "ERNIE-DSSM" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
global step 0, epoch: 1, loss: 0.62319, speed: 15.16 step/s
Evaluation precision: 0.29912, recall: 0.96226, F1: 0.45638
best F1 performence has been updated: 0.00000 --> 0.45638
global step 10, epoch: 1, loss: 0.40931, speed: 3.64 step/s
global step 20, epoch: 1, loss: 0.36969, speed: 3.69 step/s
global step 30, epoch: 1, loss: 0.33927, speed: 3.69 step/s
global step 40, epoch: 1, loss: 0.31732, speed: 3.70 step/s
global step 50, epoch: 1, loss: 0.30996, speed: 3.68 step/s
...
```

在 `logs/comment_classify` 文件下将会保存训练曲线图：

<img src='assets/dssm_train_log.png'></img>

#### 3.2.2 模型推理

和单塔模型不一样的是，双塔模型可以事先计算所有候选类别的Embedding，当新来一个句子时，只需计算新句子的Embedding，并通过余弦相似度找到最优解即可。

因此，在推理之前，我们需要提前计算所有类别的Embedding并保存。

> 类别Embedding计算

运行 `get_embedding.py` 文件以计算对应类别embedding并存放到本地：

```python
...
text_file = 'data/comment_classify/types_desc.txt'                       # 候选文本存放地址
output_file = 'embeddings/comment_classify/dssm_type_embeddings.json'    # embedding存放地址

device = 'cuda:0'                                                        # 指定GPU设备
model_type = 'dssm'                                                      # 使用DSSM还是Sentence Transformer
saved_model_path = './checkpoints/comment_classify/dssm/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()
...
```

其中，所有需要预先计算的内容都存放在 `types_desc.txt` 文件中。

文件用 `\t` 分隔，分别代表 `类别id`、`类别名称`、`类别描述`：

```txt
0	水果	指多汁且主要味觉为甜味和酸味，可食用的植物果实。
1	洗浴	洗浴用品。
2	平板	也叫便携式电脑，是一种小型、方便携带的个人电脑，以触摸屏作为基本的输入设备。
...
```

执行 `python get_embeddings.py` 命令后，会在代码中设置的embedding存放地址中找到对应的embedding文件：

```json
{
    "0": {"label": "水果", "text": "水果：指多汁且主要味觉为甜味和酸味，可食用的植物果实。", "embedding": [0.3363891839981079, -0.8757723569869995, -0.4140555262565613, 0.8288457989692688, -0.8255823850631714, 0.9906797409057617, -0.9985526204109192, 0.9907819032669067, -0.9326567649841309, -0.9372553825378418, 0.11966298520565033, -0.7452883720397949,...]},
    "1": ...,
    ...
}
```

> 模型推理

完成预计算后，接下来就可以开始推理了。

我们构建一条新评论：`这个破笔记本卡的不要不要的，差评`。

运行 `python inference_dssm.py`，得到下面结果：

```python
[
    ('平板', 0.9515482187271118),
    ('电脑', 0.8216977119445801),
    ('洗浴', 0.12220608443021774),
    ('衣服', 0.1199738010764122),
    ('手机', 0.07764233648777008),
    ('酒店', 0.044791921973228455),
    ('水果', -0.050112202763557434),
    ('电器', -0.07554933428764343),
    ('书籍', -0.08481660485267639),
    ('蒙牛', -0.16164332628250122)
]
```
函数将输出（类别，余弦相似度）的二元组，并按照相似度做倒排（相似度取值范围：[-1, 1]）。


### 3.3 Sentence Transformer（双塔）

#### 3.3.1 模型训练

修改训练脚本 `train_sentence_transformer.sh` 里的对应参数, 开启模型训练：

```sh
python train_sentence_transformer.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify/sentence_transformer" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "Sentence-Ernie" \
    --batch_size 8 \
    --max_seq_len 256 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"
```

正确开启训练后，终端会打印以下信息：

```sh
...
Evaluation precision: 0.81928, recall: 0.64151, F1: 0.71958
best F1 performence has been updated: 0.46120 --> 0.71958
global step 260, epoch: 2, loss: 0.58730, speed: 3.53 step/s
global step 270, epoch: 2, loss: 0.58171, speed: 3.55 step/s
global step 280, epoch: 2, loss: 0.57529, speed: 3.48 step/s
global step 290, epoch: 2, loss: 0.56687, speed: 3.55 step/s
global step 300, epoch: 2, loss: 0.56033, speed: 3.55 step/s
...
```

在 `logs/comment_classify` 文件下将会保存训练曲线图：

<img src='assets/sentence_transformer_train_log.png'></img>

#### 3.2.2 模型推理

Sentence Transformer 同样也是双塔模型，因此我们需要事先计算所有候选文本的embedding值。

> 类别Embedding计算

运行 `get_embedding.py` 文件以计算对应类别embedding并存放到本地：

```python
...
text_file = 'data/comment_classify/types_desc.txt'                       # 候选文本存放地址
output_file = 'embeddings/comment_classify/sentence_transformer_type_embeddings.json'    # embedding存放地址

device = 'cuda:0'                                                        # 指定GPU设备
model_type = 'sentence_transformer'                                                      # 使用DSSM还是Sentence Transformer
saved_model_path = './checkpoints/comment_classify/sentence_transformer/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()
...
```

其中，所有需要预先计算的内容都存放在 `types_desc.txt` 文件中。

文件用 `\t` 分隔，分别代表 `类别id`、`类别名称`、`类别描述`：

```txt
0	水果	指多汁且主要味觉为甜味和酸味，可食用的植物果实。
1	洗浴	洗浴用品。
2	平板	也叫便携式电脑，是一种小型、方便携带的个人电脑，以触摸屏作为基本的输入设备。
...
```

执行 `python get_embeddings.py` 命令后，会在代码中设置的embedding存放地址中找到对应的embedding文件：

```json
{
    "0": {"label": "水果", "text": "水果：指多汁且主要味觉为甜味和酸味，可食用的植物果实。", "embedding": [0.32447007298469543, -1.0908259153366089, -0.14340722560882568, 0.058471400290727615, -0.33798110485076904, -0.050156619399785995, 0.041511114686727524, 0.671889066696167, 0.2313404232263565, 1.3200652599334717, -1.10829496383667, 0.4710233509540558, -0.08577515184879303, -0.41730815172195435, -0.1956728845834732, 0.05548520386219025, ...]}
    "1": ...,
    ...
}
```

> 模型推理

完成预计算后，接下来就可以开始推理了。

我们构建一条新评论：`这个破笔记本卡的不要不要的，差评`。

运行 `python inference_sentence_transformer.py`，函数会输出所有类别里「匹配通过」的类别及其匹配值，得到下面结果：

```python
Used 0.5233056545257568s.
[
    ('平板', 1.136274814605713), 
    ('电脑', 0.8851938247680664)
]
```
函数将输出（匹配通过的类别，匹配值）的二元组，并按照匹配值（越大则越匹配）做倒排。
