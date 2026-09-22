# AR LLM Token Probability Learning

这是一个用于学习自回归语言模型 Token 预测与概率分析的实验项目，对应老师布置的实验作业。

核心学习链路：

```text
文本 -> Tokenizer -> Token IDs -> logits -> 概率
     -> next-token prediction -> 序列评分 -> NLL / Perplexity
```

## 项目内容

- `assignment/`：作业文件
- `lessons/`：分步骤学习笔记
- `docs/`：技术问题与回答
- `experiments/`：实验代码
- `results/`：实验结果
- `models/`：Tokenizer 和模型配置
- `tools/`：模型文件下载脚本

## 运行

安装 PyTorch 和 Transformers 后，在项目根目录运行：

```bash
python experiments/01_tokenizer_experiment.py
```

其余模型概率实验需要本地 GPT-2 权重。下载方法见 [models/README.md](models/README.md)，然后运行相应的实验脚本。

## 学习进度

- 自回归模型与 next-token prediction
- Token、Token ID 和 Tokenizer
- logits、softmax probability 和 log probability
- Top-k prediction
- 多 Token 候选序列概率
- Token-level scoring、NLL 和 Perplexity
- 文本概率比较

## 说明

模型权重未上传到仓库。代码、学习笔记、实验结果、作业文件和技术问答均已整理在项目中。
