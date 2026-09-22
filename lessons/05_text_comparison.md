# Lesson 05：比较两段文本的 NLL 和困惑度

## 比较原则

比较文本时要固定：

```text
同一个模型
同一个 Tokenizer
同样的评分规则
```

如果两段文本长度不同，不应该只比较 total log probability，因为更长的文本通常会累积更多负的 log probability。

这时应主要比较：

```text
NLL
perplexity
```

本次选择两句 Token 数量相同的文本：

```text
The capital of France is Paris.
The capital of France is London.
```

## 实验结果

```text
Paris 句子：
total log probability = -22.7895946503
NLL                    = 3.7982656956
perplexity             = 44.6237262186
```

```text
London 句子：
total log probability = -25.5244121552
NLL                    = 4.2540688515
perplexity             = 70.3912419616
```

## 解释

对于这个 GPT-2 模型和这个上下文：

```text
Paris 句子的 NLL 更低
Paris 句子的 perplexity 更低
```

因此，模型认为 `Paris` 这段文本比 `London` 更符合它学到的语言模式。

但这不等于模型提供了事实验证，也不能把概率直接解释为“事实正确率”。它只说明在当前模型、Tokenizer 和上下文下，模型更偏好哪种 Token 序列。

