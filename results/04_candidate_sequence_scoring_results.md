# GPT-2 Candidate Sequence Scoring 实验结果

## Prompt

```text
The capital of France is
```

Prompt Tokenization：

```text
['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
```

## 带前导空格的候选词

| Candidate | Tokens | Token IDs | Complete probability | Complete log probability |
|---|---|---|---:|---:|
| ` Paris` | `['ĠParis']` | `[6342]` | 0.032245340609 | -3.4343817234 |
| ` London` | `['ĠLondon']` | `[3576]` | 0.002118067086 | -6.1572513580 |
| ` Berlin` | `['ĠBerlin']` | `[11307]` | 0.001585415199 | -6.4469089508 |

这些候选各自只有一个 Token，因此完整候选概率就是第一个 Token 的概率。

## 不带前导空格的候选词

| Candidate | Tokens | Token IDs | Complete probability | Complete log probability |
|---|---|---|---:|---:|
| `Paris` | `['Paris']` | `[40313]` | 0.000003755015 | -12.4924182892 |
| `London` | `['London']` | `[23421]` | 0.000000318824 | -14.9586277008 |
| `Berlin` | `['Ber', 'lin']` | `[24814, 2815]` | 0.000000111236 | -16.0116138458 |

`Berlin` 的逐 Token 结果为：

| Token | Token ID | Conditional probability | Log probability |
|---|---:|---:|---:|
| `Ber` | 24814 | 0.0000001242 | -15.9015750885 |
| `lin` | 2815 | 0.8957989216 | -0.1100392863 |

因此：

```text
Berlin 的完整 log probability
= -15.9015750885 + (-0.1100392863)
= -16.0116138458
```

## 结论

1. `" Berlin"` 是一个 Token，可以直接读取概率；
2. `"Berlin"` 被拆成两个 Token，需要计算两个连续的条件概率；
3. 多 Token 词的完整概率是逐 Token 概率的乘积；
4. 多 Token 词的完整 log probability 是逐 Token log probability 的加和；
5. 比较不同长度文本时，需要注意长度惩罚，必要时比较平均 log probability 或 NLL。

