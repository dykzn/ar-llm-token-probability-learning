# GPT-2 Token-level Scoring 实验结果

## 实验设置

- 模型：GPT-2
- 输入文本：`The capital of France is Paris.`
- Tokenizer：GPT-2
- 评分方式：不添加 BOS，忽略第一个 Token，评分其余 `T - 1` 个 Token

## Tokenization

```text
Tokens:
['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis', 'ĠParis', '.']

Token IDs:
[464, 3139, 286, 4881, 318, 6342, 13]
```

## 逐 Token 结果

| Position | Context | Target Token | Token ID | Probability | Log probability |
|---:|---|---|---:|---:|---:|
| 1 | `The` | ` capital` | 3139 | 0.0001151078 | -9.0696411133 |
| 2 | `The capital` | ` of` | 286 | 0.1642125845 | -1.8065934181 |
| 3 | `The capital of` | ` France` | 4881 | 0.0065395669 | -5.0298843384 |
| 4 | `The capital of France` | ` is` | 318 | 0.1216527596 | -2.1065845490 |
| 5 | `The capital of France is` | ` Paris` | 6342 | 0.0322453417 | -3.4343817234 |
| 6 | `The capital of France is Paris` | `.` | 13 | 0.2611891925 | -1.3425102234 |

## 整段文本指标

```text
total_log_probability = -22.7895946503
nll                    = 3.7982656956
perplexity             = 44.6237262186
```

## 如何理解结果

### 1. `Paris` 的概率来自正确的前缀

表中 `Paris` 的一行是：

```text
Context：The capital of France is
Target： Paris
Probability：0.0322453417
```

它使用的是“看到 `is` 之后”的预测结果，而没有把 `Paris` 自己放进上下文。

### 2. 每一行只评分一个目标 Token

例如第一行：

```text
The →  capital
```

第二行：

```text
The capital →  of
```

模型一次前向计算可以同时得到所有位置的预测，但每个位置仍然只使用它之前的上下文。

### 3. 总 log probability 不适合直接比较不同长度文本

`total_log_probability` 是所有 Token 的 log probability 之和，文本越长通常越负。比较不同长度文本时，应优先比较：

```text
NLL 或 perplexity
```

