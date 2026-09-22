# GPT-2 Next-token Prediction 实验结果

## 实验设置

- 模型：GPT-2
- 权重：本地 `models/gpt2/model.safetensors`
- Transformers：4.57.1
- PyTorch：2.7.0+cu128
- 设备：CPU 推理
- 输入文本：`The capital of France is`

## 输入 Tokenization

```text
Tokens:
['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']

Token IDs:
[464, 3139, 286, 4881, 318]
```

## 模型输出形状

```text
logits shape: (1, 5, 50257)
last logits shape: (50257,)
```

最后一个位置的 50257 个 logits 用来预测下一个 Token。

## Top-10 next-token predictions

| Rank | Token 内部表示 | 解码文本 | Token ID | Probability | Log probability |
|---:|---|---|---:|---:|---:|
| 1 | `Ġthe` | ` the` | 262 | 0.0845910311 | -2.4699270725 |
| 2 | `Ġnow` | ` now` | 783 | 0.0479469188 | -3.0376608372 |
| 3 | `Ġa` | ` a` | 257 | 0.0461597294 | -3.0756475925 |
| 4 | `ĠFrance` | ` France` | 4881 | 0.0323759913 | -3.4303381443 |
| 5 | `ĠParis` | ` Paris` | 6342 | 0.0322453417 | -3.4343817234 |
| 6 | `Ġin` | ` in` | 287 | 0.0266409107 | -3.6253073215 |
| 7 | `Ġalso` | ` also` | 635 | 0.0264055729 | -3.6341803074 |
| 8 | `Ġnot` | ` not` | 407 | 0.0238288883 | -3.7368566990 |
| 9 | `Ġhome` | ` home` | 1363 | 0.0233487468 | -3.7572119236 |
| 10 | `Ġstill` | ` still` | 991 | 0.0155076450 | -4.1664218903 |

## 候选 Token 的第一个 Token 概率

下面的结果只比较每个候选文本的第一个 Token。对于被拆成多个 Token 的候选文本，这还不是完整的词概率。

| Candidate | Tokens | Token IDs | First probability | First log probability |
|---|---|---|---:|---:|
| ` Paris` | `['ĠParis']` | `[6342]` | 0.0322453417 | -3.4343817234 |
| ` London` | `['ĠLondon']` | `[3576]` | 0.0021180667 | -6.1572513580 |
| ` Berlin` | `['ĠBerlin']` | `[11307]` | 0.0015854150 | -6.4469089508 |
| `Paris` | `['Paris']` | `[40313]` | 0.0000037550 | -12.4924182892 |
| `London` | `['London']` | `[23421]` | 0.0000003188 | -14.9586277008 |
| `Berlin` | `['Ber', 'lin']` | `[24814, 2815]` | 0.0000001242 | -15.9015750885 |

## 结果解释

### 1. GPT-2 的最高概率不是 `Paris`

对于这个具体的 GPT-2 模型和上下文，Top-1 是：

```text
' the'，概率约为 0.0846
```

` Paris` 排名第 5，概率约为 0.0322。

这并不表示 GPT-2 认为法国首都是别的城市，而是说明模型的 next-token 分布受到模型规模、上下文形式和训练语料影响。模型概率不能直接等同于事实正确率。

### 2. 前导空格会显著影响概率

带空格的候选：

```text
' Paris' -> Token ID 6342 -> 概率约 0.0322
```

不带空格的候选：

```text
'Paris' -> Token ID 40313 -> 概率约 0.0000038
```

在句子中预测一个新单词时，通常应该比较带前导空格的形式。

### 3. 多 Token 候选不能只看第一个 Token

`Berlin` 被拆成：

```text
['Ber', 'lin']
```

因此完整的 `Berlin` 概率还需要继续计算 `lin` 在 `Ber` 后面的条件概率。后续学习 Token-level scoring 时会完成这个计算。

