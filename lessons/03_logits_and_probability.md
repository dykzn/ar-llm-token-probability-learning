# Lesson 03：从 logits 到 Token probability

## 模型输出的结构

对于文本：

```text
The capital of France is
```

GPT-2 Tokenizer 得到 5 个 Token：

```text
['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
```

对应的输入 ID 是：

```text
[464, 3139, 286, 4881, 318]
```

GPT-2 模型输出的 logits 形状是：

```text
[1, 5, 50257]
```

含义是：

- `1`：输入了 1 个样本；
- `5`：输入文本有 5 个 Token；
- `50257`：词表中有 50257 个候选 Token。

最后一个位置的 logits 形状是：

```text
[50257]
```

它表示模型对“下一个 Token”的 50257 个原始分数。

## logits、probability 和 log probability

logits 是模型产生的原始分数，不能直接当成概率。

经过 softmax 后，得到每个候选 Token 的 probability。概率越大，表示模型在当前上下文下越倾向于选择这个 Token。

对 probability 取自然对数后得到 log probability。概率越大，log probability 越接近 0；概率越小，log probability 越负。

## 本次实验的实际观察

GPT-2 给出的 Top-1 是：

```text
' the'，概率约为 0.0846
```

而：

```text
' Paris'，概率约为 0.0322
```

这说明：

1. GPT-2 并没有把 `Paris` 排在第一位；
2. 模型的概率表示语言模型偏好，不是事实正确率；
3. 模型、上下文、Tokenization 和训练数据都会影响概率；
4. ` Paris` 和 `Paris` 是不同候选 Token，概率差异很大。

