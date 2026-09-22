# GPT-2 Text Comparison 实验结果

## 比较文本

```text
The capital of France is Paris.
The capital of France is London.
```

两句话都包含 7 个 Token，并且都评分 6 个 Token。

## 结果

| Text ending | Total log probability | NLL | Perplexity |
|---|---:|---:|---:|
| `Paris.` | -22.7895946503 | 3.7982656956 | 44.6237262186 |
| `London.` | -25.5244121552 | 4.2540688515 | 70.3912419616 |

## 结论

在当前 GPT-2 模型和上下文下：

```text
Paris 句子的 NLL 更低，困惑度也更低。
```

这表示模型更偏好 `Paris` 这条 Token 序列。

该结果不能直接当作事实正确率或事实验证，只能说明模型在当前条件下的语言概率分布。

