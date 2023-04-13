# Part Ⅱ

该部分基于  [FLAVA: A Foundational Language And Vision Alignment Model](https://github.com/facebookresearch/multimodal/tree/main/examples/flava) 。



## Requirements

### Environment

按照以下步骤创建环境：

```
git clone https://github.com/facebookresearch/multimodal.git
cd multimodal
pip install -e .
cd examples
pip install -r flava/requirements.txt
```

注意 requirements.txt 中的datasets version 应为2.6.2及以上，torch版本与torch-lightning版本都应与服务器当前的CUDA版本对应。

### 注册[HuggingFace](https://huggingface.co/join)账号（可选）

若要使用 ImageNet 数据集，则须先在 [HuggingFace](https://huggingface.co/join) 创建一个帐户。 

创建帐户并确认电子邮件后，登录，单击profile，然后 Settings -> Access Tokens。 创建一个具有 READ 访问权限的新token并复制，然后在您的终端中运行

```
huggingface-cli login
```

并粘贴token。 

可在 *~/.huggingface/token* 路径中发现该token。 记得在[数据集页面](https://huggingface.co/datasets/imagenet-1k) 接受条款。




## Training

使用GPU:
```
python -m flava.train config=flava/configs/pretraining/debug.yaml model.pretrained=True
```

例如:

```
python -m flava.train config=flava/configs/pretraining/debug_all_transfer_market.yaml
```

model.pretrained可选，=True即指使用预训练模型。



## Finetuning

针对下游的ReID任务进行微调，例如：

```
python -m flava.finetune config=flava/configs/finetuning/debug_real_market.yaml model.pretrained=True
```



## 注意

1. 所用的数据集在第一次加载时会耗费一定的时间生成.arrow文件，但之后的数据加载则会快很多。

   若要使用不在HuggingFace库中的数据集，则需自行配置数据集的加载格式，举个例子详见*multimodal-main/examples/transfer_market* 文件夹。

2. 训练的log文件自动存于*multimodal-main/examples/lightning_logs* 文件夹，可利用 tensorboardX 连接本地浏览器直接可视化。
