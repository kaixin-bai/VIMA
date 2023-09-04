# VIMA: General Robot Manipulation with Multimodal Prompts
## ICML 2023
<div align="center">

[[Website]](https://vimalabs.github.io/)
[[arXiv]](https://arxiv.org/abs/2210.03094)
[[PDF]](https://vimalabs.github.io/assets/vima_paper.pdf)
[[Pretrained Models]](#Pretrained-Models)
[[Baselines Implementation]](#Baselines-Implementation)
[[VIMA-Bench]](https://github.com/vimalabs/VimaBench)
[[Training Data]](https://huggingface.co/datasets/VIMA/VIMA-Data)
[[Model Card]](model-card.md)

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://github.com/vimalabs/VIMA)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/vimalabs/VIMA)](https://github.com/vimalabs/VIMA/blob/main/LICENSE)
______________________________________________________________________
![](images/pull.png)
</div>

Prompt-based learning has emerged as a successful paradigm in natural language processing, where a single general-purpose language model can be instructed to perform any task specified by input prompts. However, different robotics tasks are still tackled by specialized models. This work shows that we can express a wide spectrum of robot manipulation tasks with *multimodal prompts*, interleaving textual and visual tokens.
We introduce VIMA (**Vi**suo**M**otor **A**ttention agent), a novel scalable multi-task robot learner with a uniform sequence IO interface achieved through multimodal prompts. The architecture follows the encoder-decoder transformer design proven to be effective and scalable in NLP. VIMA encodes an input sequence of interleaving textual and visual prompt tokens with a [pretrained](https://www.deepmind.com/publications/multimodal-few-shot-learning-with-frozen-language-models) [language model](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html), and decodes robot control actions autoregressively for each environment interaction step. The transformer decoder is conditioned on the prompt via cross-attention layers that alternate with the usual causal self-attention. Instead of operating on raw pixels, VIMA adopts an object-centric approach. We parse all images in the prompt or observation into objects by [off-the-shelf detectors](https://arxiv.org/abs/1703.06870), and flatten them into sequences of object tokens. All these design choices combined deliver a conceptually simple architecture with strong model and data scaling properties.

In this repo, we provide VIMA model code, pre-trained checkpoints covering a spectrum of model sizes, and demo and eval scripts. This codebase is under [MIT License](LICENSE).

# Installation
VIMA requires Python ≥ 3.9. We have tested on Ubuntu 20.04. Installing VIMA codebase is as simple as:

```bash
pip install git+https://github.com/vimalabs/VIMA
```

# Pretrained Models
We host pretrained models covering a spectrum of model capacity on [Hugging Face](https://huggingface.co/VIMA/VIMA). Download links are listed below. The mask R-CNN model can be found [here](https://huggingface.co/VIMA/VIMA/resolve/main/mask_rcnn.pth).

| [200M](https://huggingface.co/VIMA/VIMA/resolve/main/200M.ckpt) | [92M](https://huggingface.co/VIMA/VIMA/resolve/main/92M.ckpt) | [43M](https://huggingface.co/VIMA/VIMA/resolve/main/43M.ckpt) | [20M](https://huggingface.co/VIMA/VIMA/resolve/main/20M.ckpt) | [9M](https://huggingface.co/VIMA/VIMA/resolve/main/9M.ckpt) | [4M](https://huggingface.co/VIMA/VIMA/resolve/main/4M.ckpt) | [2M](https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt)    |
|-----------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|-----|

# Baselines Implementation
Because there is no prior method that works out of the box with our multimodal prompting setup, we make our best effort to select a number of representative transformer-based agent architectures as baselines, and re-interpret them to be compatible with VIMA-Bench. They include ```VIMA-Gato```, ```VIMA-Flamingo```, and ```VIMA-GPT```. Their implementation can be found in the ```policy``` folder.

# Demo
To run the live demonstration, first follow the [instruction](https://github.com/vimalabs/VimaBench/tree/main#installation) to install [VIMA-Bench](https://github.com/vimalabs/VimaBench).Then we can run a live demo through

```bash
python3 scripts/example.py --ckpt={ckpt_path} --device={device} --partition={eval_level} --task={task}
```

Here `eval_level` means one out of four evaluation levels and can be chosen from `placement_generalization`, `combinatorial_generalization`, `novel_object_generalization`, and `novel_task_generalization`. `task` means a specific task template. Please refer to [task suite](https://github.com/vimalabs/VimaBench/tree/main#task-suite) and [benchmark](https://github.com/vimalabs/VimaBench/tree/main#evaluation-benchmark) for more details.

After running the above command, we should see a PyBullet GUI pop up, alongside a small window showing the multimodal prompt. Then a robot arm should move to complete the corresponding task. Note that this demo may not work on headless machines since the PyBullet GUI requires a display.

# Paper and Citation

Our paper is posted on [arXiv](https://arxiv.org/abs/2210.03094). If you find our work useful, please consider citing us! 

```bibtex
@inproceedings{jiang2023vima,
  title     = {VIMA: General Robot Manipulation with Multimodal Prompts},
  author    = {Yunfan Jiang and Agrim Gupta and Zichen Zhang and Guanzhi Wang and Yongqiang Dou and Yanjun Chen and Li Fei-Fei and Anima Anandkumar and Yuke Zhu and Linxi Fan},
  booktitle = {Fortieth International Conference on Machine Learning},
  year      = {2023}
}
```

# 笔记
## 安装
```bash
# 创建环境
conda create -n vima python=3.9
conda activate vima

# 安装pytorch，根据本机和服务器上的测试定的版本
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

# 安装其他依赖
pip3 install dm-tree kornia einops tokenizers transformers setuptools==57.5.0

# 安装本项目，注意，在setup.py中取消了安装requirements.txt中的依赖项，我们已经提前安装过了
pip3 install .
```
**dm-tree**: 为DeepMind团队开发基于C++实现的独立轻量级树状运算库，同类产品中虽然轻量但是性能差(轻度使用场景的话无所谓)。  \
作用：将树形展开成可逆的列表结构；简单的函数式映射运算  \
相关内容：[知乎上对树形库的测试](https://zhuanlan.zhihu.com/p/467483175)  \
**kornia**: 基于PyTorch 的可微分的计算机视觉库，实现了可微的基础计算机视觉算子和可微的数据增广。  \
**einops**: 提供常用张量操作的Python包，支持NumPy、Tensorflow、PyTorch等框架，可以与这些框架有机衔接。 其功能涵盖了reshape、view、transpose和permute等操作。 其特点是可读性强、易维护，如变更轴的顺序的操作。  \
**tokenizers**: [详细教程之Tokenizer库](https://zhuanlan.zhihu.com/p/591335566) SOTA tokenizer的库。  \
**transformers**: 提供了数以千计的预训练模型，支持 100 多种语言的文本分类、信息抽取、问答、摘要、翻译、文本生成;提供了便于快速下载和使用的API，让你可以把预训练模型用在给定文本、在你的数据集上微调.

## 预训练模型下载
我们将所有的预训练模型都下载至以下文件夹中`/data/net/dl_data/ProjectDatasets_bkx/VIMA_pretrained_models`
```bash
cd /data/net/dl_data/ProjectDatasets_bkx/VIMA_pretrained_models/
# 下载vima中使用的maskrcnn的预训练模型
wget https://huggingface.co/VIMA/VIMA/resolve/main/mask_rcnn.pth
# 下载vima的预训练模型 - 200M
wget https://huggingface.co/VIMA/VIMA/resolve/main/200M.ckpt
# 下载vima的预训练模型 - 92M
wget https://huggingface.co/VIMA/VIMA/resolve/main/92M.ckpt
# 下载vima的预训练模型 - 43M
wget https://huggingface.co/VIMA/VIMA/resolve/main/43M.ckpt
# 下载vima的预训练模型 - 20M
wget https://huggingface.co/VIMA/VIMA/resolve/main/20M.ckpt
# 下载vima的预训练模型 - 9M
wget https://huggingface.co/VIMA/VIMA/resolve/main/9M.ckpt
# 下载vima的预训练模型 - 4M
wget https://huggingface.co/VIMA/VIMA/resolve/main/4M.ckpt
# 下载vima的预训练模型 - 2M
wget https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt
```

## VIMA-Bench下载安装
```bash
git clone https://github.com/kaixin-bai/VIMABench.git
git checkout testing-and-notes
pip install -e .
```

## 效果测试
```bash
python3 scripts/example.py --ckpt=/data/net/dl_data/ProjectDatasets_bkx/VIMA_pretrained_models/20M.ckpt --device=cpu --partition=placement_generalization --task=visual_manipulation
```


# 遇到的问题和解决方法
```bash
pip install gpy==0.21.0
pip install git+https://github.com/openai/gym.git@v0.21.0
```
在安装gym的0.21.0版本时报错：
```bash
om/openai/gym.git@v0.21.0
Collecting git+https://github.com/openai/gym.git@v0.21.0
  Cloning https://github.com/openai/gym.git (to revision v0.21.0) to /tmp/pip-req-build-kftlr7ux
  Running command git clone --filter=blob:none --quiet https://github.com/openai/gym.git /tmp/pip-req-build-kftlr7ux

  Running command git checkout -q c755d5c35a25ab118746e2ba885894ff66fb8c43
  Resolved https://github.com/openai/gym.git to commit c755d5c35a25ab118746e2ba885894ff66fb8c43
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```
原因是由于`setuptools`高版本做的一些修改导致`pip`安装`gym`时出问题，因为`gym`已经两年没更新了(最后一次更新为2021年)。所以解决方法为对`setuptools`进行降级操作，[参考网页](https://github.com/openai/gym/issues/3176) ，如下：
```bash
# 当前版本 setuptools         68.0.0
# pip install setuptools==65.5.0  # 上面网页提供的方法，不过据说也不行
# https://blog.csdn.net/weixin_60245579/article/details/131013371
pip install setuptools==57.5.0  # csdn上提供的方法 
```

# 代码笔记
在`example_debug.py`中，`policy = create_policy_from_ckpt(cfg.ckpt, cfg.device)`进入读取预训练模型函数，其中的`policy_instance`是`VIMAPolicy`.  \
在进入到`class VIMAPolicy`之后，可以看到以下内容：
```python
self.xattn_gpt = vnn.XAttnGPT(
    embed_dim,
    n_layer=xf_n_layers,
    n_head=sattn_n_heads,
    dropout=0.1,
    xattn_n_head=xattn_n_heads,
    xattn_ff_expanding=4,
    xattn_n_positions=256,
    use_geglu=True,
)
# =============================================================================================
tokens_out = self.xattn_gpt(
    obs_action_tokens=tokens,
    prompt_tokens=prompt_token,
    prompt_mask=prompt_token_mask,
    obs_action_masks=masks.transpose(0, 1),
    obs_action_position_ids=position_ids.transpose(0, 1),
    prompt_position_ids=prompt_position_ids,
)
predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]
```
然后我们详细看一下这个`XAttnGPT`