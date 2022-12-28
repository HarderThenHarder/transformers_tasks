# Reinforcement Learning with Language Model

随着 ChatGPT 的爆火，强化学习（Reinforcement Learning）和语言生成模型（Language Model）的结合开始变得越来越受人关注。

有关 ChatGPT 的视频讲解可以参考[这里](https://www.bilibili.com/video/BV1BG4y137SH/?vd_source=0df98e40ba56afac31703b0d5dba509f#reply143954452208)。

在这个项目中，我们将通过开源项目 [trl](https://github.com/lvwerra/trl) 搭建一个通过强化学习算法（PPO）来更新语言模型（GPT-2）的几个示例，包括：

* 基于中文情感识别模型的正向评论生成机器人（No Human Reward）

* 基于人工打分的评论生成机器人（With Human Reward）


### 1. 基于中文情感识别模型的正向评论生成机器人（No Human Reward）

考虑现在我们有一个现成的语言模型（示例中选用中文的GPT2），通过一小段 `prompt`，模型能够继续生成一段文字，例如：

```python
prompt: 刚收到货，感觉有

output 1: 刚收到货，感觉有 点 不 符 合 预 期 ，不 好
output 2: 刚收到货，感觉有 挺 无 奈 的 送 货 速 度 不 太 行
...
```

我们现在希望语言模型能够学会生成「正向情感」的好评，而当前的 GPT 模型是不具备「情绪识别」能力的，如上面两个生成结果都不符合正面情绪。

为此，我们期望通过「强化学习」的方法来进化现有 GPT 模型，使其能够学会尽可能的生成「正面情感」的评论。

在强化学习中，当模型生成一个结果时，我们需要告知模型这个结果的得分（reward）是多少，即我们为模型的每一个生成结果打分，例如：


```python
output 1: 刚收到货，感觉有 点 不 符 合 预 期 ，不 好                -> 0.2 分
output 2: 刚收到货，感觉有 挺 无 奈 的 送 货 速 度 不 太 行          -> 0.1 分
output 3: 刚收到货，感觉有 些 惊 喜 于 货 物 质 量                  -> 0.9 分
...
```

如果依靠人工为每一个输出打分，这将是一个非常漫长的过程（在另一个示例中我们将实现该功能），因此，我们引入另一个「情绪识别模型」来模拟人工给出的分数。

「情绪识别模型」我们选用 transformers 中内置的 sentiment-analysis pipeline 来实现，[该模型](https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese)基于网络评论数据集训练，能够对句子进行「正向、负向」的情绪判别，如下所示：

<img src='assets/sentiment-analysis.png'>

我们利用该「情感识别模型」的判别结果（0.0~1.0）作为 GPT 生成模型的 reward，以指导 GPT 模型通过强化学习（PPO）算法进行迭代更新。

#### 1.1 训练流程

整个 PPO + GPT2 的训练流程如下所示：

1. 随机选择一个 `prompt`，如："这部电影很"

2. GPT 模型根据 `prompt` 生成答案，如："这部电影很 好 看 哦 ~ "

3. 将 GPT 的生成答案喂给「情绪识别」模型，并得到评分（reward），如：0.9

4. 利用评分（reward）对 GPT 模型进行优化。

重复该循环，直到训练结束为止。

#### 1.2 开始训练

本项目基于 `pytorch` + `transformers` 实现，运行前请安装相关依赖包：

```sh
pip install -r ../requirements.txt
```

运行训练脚本：

```sh
python ppo_sentiment_example.py
```

正常启动训练后，终端会打印如下数据：

```sh
...
epoch 0 mean-reward: 0.7271811366081238
Random Sample 5 text(s) of model output:
1. 刚收到货，感觉不 错 ， 会 冒 充 收 银 员 在 果 盘 盘 底 ， 就
2. 说实话，真的很般 般 ， 一 般 都 是 饭 点 去 ， 没 办 法 我 现
3. 说实话，真的很怪 不 得 刚 开 的 没 多 久 ， 现 在 上 海 这 个
4. 这部电影很啊 ， 所 以 ， 也 算 一 个 抛 砖 引 玉 。 昨 天
5. 这次购物总的来说体验很[SEP] ~ 满 意 谢 谢 送 货 很 快 [SEP] 为 什 么 输 出
  1%|▋                                                                                                     | 1/157 [00:55<2:23:53, 55.34s/it]epoch 1 mean-reward: 0.7439988851547241
Random Sample 5 text(s) of model output:
1. 这次购物总的来说体验很我 不 知 道 表 盘 这 是 男 人 的? 听 说 女 人
2. 这部电影很金 士 顿 鉴 定 和 暗 暗 [SEP] 正 品 。 是 正 品 这
3. 刚收到货，感觉是 有 些 人 吃 不 尽 的 名 字 ！ ~ 世 界 几 大
4. 说实话，真的很对 不 起 这 个 价 钱 ， 可 能 是 因 为 做 出 来
5. 说实话，真的很非 电 。 31. 可 说 是 食 堂 ， 没 怎 么 规 划
  1%|█▎                                                                                                    | 2/157 [01:51<2:24:31, 55.95s/it]epoch 2 mean-reward: 0.8219242691993713
...
```

其中 `mean-reward` 代表该 epoch 下模型的平均得分（来自「情绪识别模型」的反馈），`Random Sample` 代表该模型在当前 epoch 生成的句子样例。

在 `logs/PPO-Sentiment-Zh.png` 下会保存模型训练过程中的各个指标变化（包括 reward 变化曲线）：

<img src='assets/PPO-Sentiment-Zh.png'>

在模型刚开始训练的时候，GPT 会生成一些比较随机的答案，此时的平均 reward 也不会很高，会生成一些「负面」情绪的评论（如下所示）：

<img src='assets/start.jpg'>

随着训练，GPT 会慢慢学会偏向「正面」的情绪评论（如下所示）：

<img src='assets/end.jpg'>


### 2. 基于人工打分的评论生成机器人（With Human Reward）

在第一个示例中，模型的 reward 来自于另一个模型。

在该示例中，我们将制作一个平台来支持人工进行打分。

然后，我们启动标注平台：

```sh
python terminal_main.py 
```

随后我们可以在终端看到模型的生成结果，通过人工输入 reward 以迭代模型：

<img src='assets/terminal.png'>


