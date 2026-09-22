# AR LLM Token Probability Learning

这是一个用于循序学习“自回归语言模型 Token 预测与概率分析”的小项目，对应老师布置的实验作业。

项目目标不是只得到一个结果，而是理解下面这条完整链路：

```text
文本 -> Tokenizer -> Token IDs -> 模型 logits -> softmax 概率
     -> next-token prediction -> 候选序列概率 -> NLL / Perplexity
```

## 项目目录

```text
AR_LLM_Token_Probability_Learning/
├── README.md
├── .gitignore
├── assignment/                         # 老师布置的作业
├── lessons/                            # 分步骤学习笔记
├── docs/technical_qa.md                # 学习过程中的技术问答
├── experiments/                        # 可运行实验代码
├── results/                            # 实验输出和解释
├── models/                             # 配置和 tokenizer；不包含模型权重
│   ├── README.md
│   ├── gpt2-tokenizer/
│   └── gpt2/
└── tools/hfd.sh                        # 直接下载 Hugging Face 文件的脚本
```

## 学习进度

- [x] 理解自回归模型每一步预测下一个 Token
- [x] 理解 Token、Token ID 和 Tokenizer
- [x] 观察 `Paris` 与 ` Paris` 的区别
- [x] 使用语言模型得到 logits
- [x] 把 logits 转换为 probability
- [x] 完成 Top-10 next-token prediction
- [x] 完成候选 Token 的第一个 Token 概率比较
- [x] 完成多 Token 候选词的完整概率计算
- [x] 完成逐 Token scoring、NLL 和困惑度
- [x] 比较两段等长度文本的 NLL 和困惑度
- [x] 整理技术问题与回答

## 如何运行

### 1. 只运行 tokenizer 实验

这个实验不需要模型权重，也不需要 GPU：

```bash
cd /data3/dengyongkang/my_project/AR_LLM_Token_Probability_Learning
/data3/dengyongkang/.conda/envs/cosmos-policy/bin/python \
  experiments/01_tokenizer_experiment.py
```

### 2. 运行模型概率实验

`experiments/02` 到 `experiments/05` 需要本地 GPT-2 权重。请先阅读 [models/README.md](models/README.md)，使用 `hfd.sh` 下载权重；权重不会上传到 GitHub。

然后分别运行：

```bash
/data3/dengyongkang/.conda/envs/cosmos-policy/bin/python experiments/02_next_token_prediction.py
/data3/dengyongkang/.conda/envs/cosmos-policy/bin/python experiments/03_token_level_scoring.py
/data3/dengyongkang/.conda/envs/cosmos-policy/bin/python experiments/04_candidate_sequence_scoring.py
/data3/dengyongkang/.conda/envs/cosmos-policy/bin/python experiments/05_compare_texts.py
```

## 关键文档

- [学习笔记](lessons/)
- [技术问答](docs/technical_qa.md)
- [实验结果](results/)
- [作业 PDF](assignment/AR_LLM_Token_Probability_Assignment.pdf)

## 运行环境

- Python：`/data3/dengyongkang/.conda/envs/cosmos-policy/bin/python`
- PyTorch：`2.7.0+cu128`
- Transformers：`4.57.1`
- 模型：GPT-2 (`openai-community/gpt2`)

## 仓库约定

- 概念解释放到 `lessons/`。
- 学习过程中的问题和回答放到 `docs/technical_qa.md`。
- 可运行代码放到 `experiments/`。
- 运行结果放到 `results/`。
- 模型配置和 tokenizer 放到 `models/`；大型权重只在本地保存，不提交到 GitHub。
