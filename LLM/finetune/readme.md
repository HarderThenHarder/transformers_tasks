# Finetune ChatGLM-6B

LLM（Large Language Model）通常拥有大量的先验知识，使得其在许多自然语言处理任务上都有着不错的性能。

但，想要直接利用 LLM 完成一些任务会存在一些答案解析上的困难，如规范化输出格式，严格服从输入信息等。

> Zero-Shot 实验代码在 [这里](../zero-shot/readme.md)。

因此，在这个项目下我们参考 [这里](https://github.com/mymusise/ChatGLM-Tuning/tree/master) 的代码，尝试对大模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 进行 Finetune，使其能够更好的对齐我们所需要的输出格式。

<br>

## 1. 环境安装

由于 ChatGLM 需要的环境和该项目中其他实验中的环境有所不同，因此我们强烈建议您创建一个新的虚拟环境来执行该目录下的全部代码。

下面，我们将以 `Anaconda` 为例，展示如何快速搭建一个环境：

1. 创建一个虚拟环境，您可以把 `llm_env` 修改为任意你想要新建的环境名称：

```sh
conda create -n llm_env python=3.8
```

2. 激活新建虚拟环境并安装响应的依赖包：

```sh
conda activate llm_env
pip install -r requirements.txt
```

3. 安装对应版本的 `peft`：

```sh
cd peft-chatglm
python setup.py install
```

<br>

## 2. 数据集准备

在该实验中，我们将尝试使用 `信息抽取` + `文本分类` 任务的混合数据集喂给模型做 finetune，数据集在 `data/mixed_train_dataset.jsonl`。

每一条数据都分为 `context` 和 `target` 两部分：

1.  `context` 部分是接受用户的输入。

2. `target` 部分用于指定模型的输出。

在 `context` 中又包括 2 个部分：

1. Instruction：用于告知模型的具体指令，当需要一个模型同时解决多个任务时可以设定不同的 Instruction 来帮助模型判别当前应当做什么任务。

2. Input：当前用户的输入。

*  信息抽取数据示例

Instruction 部分告诉模型现在需要做「阅读理解」任务，Input 部分告知模型要抽取的句子以及输出的格式。

```json
{
    "context": "Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。\nInput: 找到句子中的三元组信息并输出成json给我:\n\n九玄珠是在纵横中文网连载的一部小说，作者是龙马。\nAnswer: ", 
    "target": "```json\n[{\"predicate\": \"连载网站\", \"object_type\": \"网站\", \"subject_type\": \"网络小说\", \"object\": \"纵横中文网\", \"subject\": \"九玄珠\"}, {\"predicate\": \"作者\", \"object_type\": \"人物\", \"subject_type\": \"图书作品\", \"object\": \"龙马\", \"subject\": \"九玄珠\"}]\n```"
}
```

*  文本分类数据示例

Instruction 部分告诉模型现在需要做「阅读理解」任务，Input 部分告知模型要抽取的句子以及输出的格式。

```json
{
    "context": "Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。\nInput: 下面句子可能是一条关于什么的评论，用列表形式回答：\n\n很不错，很新鲜，快递小哥服务很好，水果也挺甜挺脆的\nAnswer: ", 
    "target": "[\"水果\"]"
}
```

<br>

## 3. 模型训练

运行 `train.sh` 文件，根据自己 GPU 的显存调节 `batch_size`, `max_source_seq_len`, `max_target_seq_len` 参数（暂不支持多卡训练）：

```sh
python train.py \
    --train_path data/mixed_train_dataset.jsonl \
    --dev_path data/mixed_dev_dataset.jsonl \
    --lora_rank 8 \
    --batch_size 1 \
    --num_train_epochs 3 \
    --save_freq 1000 \
    --learning_rate 3e-5 \
    --logging_steps 100 \
    --max_source_seq_len 400 \
    --max_target_seq_len 300 \
    --save_dir checkpoints/ \
    --img_log_dir "log/fintune_log" \
    --img_log_name "ChatGLM Fine-Tune" \
    --device cuda:0
```

成功运行程序后，会看到如下界面：

```python
...
global step 1700 (62.82%) , epoch: 2, loss: 0.58667, speed: 1.28 step/s
global step 1800 (66.52%) , epoch: 2, loss: 0.55911, speed: 1.28 step/s
global step 1900 (70.21%) , epoch: 3, loss: 0.53448, speed: 1.28 step/s
global step 2000 (73.91%) , epoch: 3, loss: 0.51127, speed: 1.28 step/s
Model has saved at checkpoints/model_2000/chatglm-lora.pt.
Evaluation Loss: 0.12003
Min eval loss has been updated: 0.14775 --> 0.12003
Best model has saved at checkpoints/model_best/chatglm-lora.pt.
...
```

在 `log/finetune_log` 下会看到训练 loss 的曲线图：

<div align='center'><img src='assets/ChatGLM Fine-Tune.png'></div>

<br>

## 4. 模型预测

修改训练模型的存放路径，运行 `python inference.py` 以测试训练好模型的效果：

```python
...

peft_path = "checkpoints/model_best/chatglm-lora.pt"        # 训练模型存放路径
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8,
    lora_alpha=32, 
    lora_dropout=0.1
)
...
```

您也可以使用我们提供的 Playground 来进行模型效果测试：

```sh
streamlit run playground_local.py --server.port 8001
```

在浏览器中打开对应的 `机器ip:8001` 即可访问。

<div align='center'><img src='assets/playground.png'></div>


<br>

## 5. Instruction & Label 标注平台

如果您需要标注自己的数据，也可以在 Playground 中完成。

```sh
streamlit run playground_local.py --server.port 8001
```

在浏览器中打开对应的 `机器ip:8001` 即可访问。

<table>
<tr>
<td><img src="assets/label1.png" border=0></td>
<td><img src="assets/label2.png" border=0></td>
</tr>
</table>