# 技术问题与回答

这份文档记录学习过程中提出的关键问题。公式尽量使用普通文本和代码块表示，避免复杂的数学排版。

## 1. 自回归语言模型到底在做什么？

自回归语言模型每次只预测“下一个 Token”。它看到当前前缀后，输出整个词表上每个候选 Token 的分数。

例如：

```text
输入：The capital of France is
目标：预测下一个 Token
```

模型并不是直接输出一个确定的完整答案，而是先输出约 50,257 个候选 Token 的分数，再从中得到概率。

## 2. Token、Token ID、Tokenizer 分别是什么？

- Token：文本被切分后的片段。
- Token ID：词表中这个 Token 对应的整数编号。
- Tokenizer：负责“文本 -> Token -> Token ID”，也负责反向解码。

例如在 GPT-2 tokenizer 中，` Paris` 和 `Paris` 可能不是同一个 Token：

```text
"Paris"  -> ["Paris"]  -> [40313]
" Paris" -> ["ĠParis"] -> [6342]
```

这里的 `Ġ` 是 GPT-2 tokenizer 用来表示前导空格的标记。因此，英文候选词通常要注意前面是否带空格。

## 3. `The capital of France is` 的具体模型输出是什么？

输入首先被编码为：

```text
Tokens:    ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
Token IDs: [464, 3139, 286, 4881, 318]
```

GPT-2 的原始输出形状是：

```text
outputs.logits.shape = (1, 5, 50257)
```

这三个维度分别表示：

```text
1       = batch size
5       = 输入 Token 数量
50257   = 词表大小
```

每个输入位置都有一个长度为 50,257 的 logits 向量。预测整个输入之后的下一个 Token 时，取最后一个位置：

```python
last_logits = outputs.logits[0, -1, :]
```

GPT-2 在本次运行中的部分结果如下：

| 候选 Token | Token ID | 概率 |
|---|---:|---:|
| ` the` | 262 | 0.084591 |
| ` now` | 783 | 0.047947 |
| ` a` | 257 | 0.046160 |
| ` France` | 4881 | 0.032376 |
| ` Paris` | 6342 | 0.032245 |
| ` in` | 287 | 0.026641 |

所以，“模型的具体输出”不是只有 `Paris`，而是对整个词表给出一组分数；`Paris` 只是其中一个候选。

## 4. `last_logits` 具体是什么？为什么取最后一个位置？

`last_logits` 是模型对“下一个 Token”预测的原始分数向量：

```text
last_logits.shape = (50257,)
```

输入有 5 个 Token 时，模型会给出 5 个位置的输出。因果语言模型中，第 5 个位置对应“看完输入后预测下一个 Token”，所以要取：

```python
outputs.logits[0, -1, :]
```

其中：

```text
0       = 第 0 个样本
-1      = 最后一个输入位置
:       = 取这个位置上的全部词表分数
```

## 5. 为什么 `softmax` 使用 `dim=-1`？

模型输出的 logits 形状通常是：

```text
(batch_size, sequence_length, vocabulary_size)
```

最后一维是词表维度。我们要让同一个位置上的所有候选 Token 概率加起来等于 1，因此要沿最后一维做 softmax：

```python
probabilities = torch.softmax(last_logits, dim=-1)
```

`dim=-1` 表示“最后一个维度”。它的好处是无论前面的 batch 或 sequence 维度如何变化，都仍然指向词表维度。

## 6. Logits、概率和 log probability 有什么区别？

```text
logits          = 模型直接输出的未归一化分数
probability     = logits 经过 softmax 后的概率
log probability = 概率取自然对数后的结果
```

logits 可以是任意实数，不需要加起来等于 1。softmax 后才得到合法的概率分布。

实践中更推荐直接使用：

```python
log_probs = torch.log_softmax(logits, dim=-1)
```

而不是先 softmax 再取 log，因为 `log_softmax` 数值上更稳定。

## 7. logits 的第 `i` 个位置预测哪个 Token？

第 `i` 个位置的 logits 预测第 `i + 1` 个 Token：

```text
logits[:, 0, :] -> 预测 input_ids[:, 1]
logits[:, 1, :] -> 预测 input_ids[:, 2]
logits[:, 2, :] -> 预测 input_ids[:, 3]
```

因此逐 Token scoring 时通常这样对齐：

```python
prediction_logits = outputs.logits[:, :-1, :]
target_ids = input_ids[:, 1:]
```

两者的 sequence length 必须一致。

## 8. 为什么不会发生未来信息泄露？

GPT-2 使用因果注意力掩码。位置 `i` 只能关注自己和前面的 Token，不能看到后面的 Token。

例如在预测句子中的某个 Token 时，模型可以使用：

```text
真实前缀：A B C
```

但不能提前看到后面的目标 Token 或未来文本。

训练或评分时使用真实前缀叫 teacher forcing。它并不等于看到了未来；它只表示每一步都用正确的历史前缀来评估下一步预测。

## 9. 为什么候选词前面是否有空格很重要？

模型预测的是 Token，不一定是完整单词。下面两个字符串在 tokenizer 中可能对应不同 Token：

```text
" Paris" -> ["ĠParis"]
"Paris"  -> ["Paris"]
```

在输入 `The capital of France is` 后，真正自然的续写通常是带前导空格的 ` Paris`。本次 GPT-2 结果：

```text
P(" Paris" | prompt) ≈ 0.032245
P("Paris"  | prompt) ≈ 0.0000038
```

这不表示模型认为两个字符串的语义完全不同，而是它们处在不同的 Token 边界位置。

## 10. 多 Token 候选词应该怎样计算概率？

如果候选文本被切成多个 Token，不能只看第一个 Token，也不能把每个 Token 都在同一个 prompt 下单独查询。

例如候选 `Berlin` 被切成：

```text
['Ber', 'lin']
```

正确过程是：

```text
P("Ber lin" | prompt)
= P("Ber" | prompt)
  * P("lin" | prompt + "Ber")
```

在 log probability 中，乘法变成加法：

```text
log P(candidate | prompt)
= log P(token_1 | prompt)
  + log P(token_2 | prompt + token_1)
```

本次运行中：

```text
P(" Paris"  | prompt) ≈ 0.032245
P(" London" | prompt) ≈ 0.002118
P(" Berlin" | prompt) ≈ 0.000000111
```

`Berlin` 的第一个 Token 本身已经很不常见；第二个 Token `lin` 在已有 `Ber` 后的条件概率很高，但完整候选概率仍然要把前后概率相乘。

## 11. NLL 是什么？为什么不能总是只看最后一个词？

对于完整文本评分，NLL 会累积每个目标 Token 的负 log probability：

```text
NLL = - 平均的 token log probability
```

例如：

```text
The capital of France is Paris.
```

评分的是：

```text
P(" capital" | "The")
P(" of"     | "The capital")
P(" France" | "The capital of")
P(" is"     | "The capital of France")
P(" Paris"  | "The capital of France is")
P("."       | "The capital of France is Paris")
```

每一行的上下文都不同，所以这些概率通常不一样。完整文本的 NLL 不能只用最后一个词替代。

但如果问题明确是“在固定 prompt 后，比较下一个词是 Paris 还是 London”，那么只比较：

```text
P(" Paris"  | prompt)
P(" London" | prompt)
```

这两种做法对应不同目标：

```text
单步 next-token prediction -> 只研究最后一步
完整 sequence scoring       -> 研究整段文本的语言连贯性
```

## 12. NLL 和 Perplexity 怎么理解？

本项目中：

```text
NLL = - 平均 log probability
Perplexity = exp(NLL)
```

数值越低，表示模型对这段文本平均越有把握。

本次等长度比较结果：

| 文本 | Total log probability | NLL | Perplexity |
|---|---:|---:|---:|
| `... is Paris.` | -22.7896 | 3.7983 | 44.6237 |
| `... is London.` | -25.5244 | 4.2541 | 70.3912 |

在这个 GPT-2 和这个评分设置下，第一段文本的模型概率更高。

比较不同文本时要注意 Token 数量和 tokenizer 切分方式。总 log probability 会受文本长度影响，因此通常还要看平均 NLL 或 Perplexity。

## 13. 概率更高是否代表事实一定正确？

不代表。

语言模型概率主要反映它在训练数据中学到的语言模式、词语共现关系和上下文延续习惯。模型可能记住某些事实，但：

- 高概率不等于事实正确；
- 低概率也不一定等于事实错误；
- 模型可能生成语法自然但内容错误的句子；
- 概率通常也没有完美校准为“正确率”。

因此要区分两个问题：

```text
语言模型评分：这段文本像不像训练数据中的自然延续？
事实核验：这段文本描述的内容是否真实？
```

本项目主要学习前一个问题。

## 14. Temperature 会改变什么？

temperature 作用在 logits 到概率的转换阶段：

```python
scaled_logits = logits / temperature
probabilities = torch.softmax(scaled_logits, dim=-1)
```

直观上：

```text
temperature < 1 -> 分布更尖锐，更偏向高分 Token
temperature = 1 -> 原始模型概率
temperature > 1 -> 分布更平坦，低概率 Token 更容易出现
```

temperature 不会改变模型前向计算得到的原始 logits，也不会改变 Tokenizer。

## 15. 为什么代码里经常使用 `gather`？

模型会为每个位置输出整个词表的 log probability，但我们只需要取“真实目标 Token”对应的那一个值。

```python
token_log_probs = log_probs.gather(
    dim=-1,
    index=target_ids.unsqueeze(-1),
).squeeze(-1)
```

`gather` 的作用就是按照 `target_ids` 指定的词表索引，取出每个位置的目标 Token 概率。

## 16. 这套实验代码的输入和输出关系

```text
01_tokenizer_experiment.py
    观察文本如何切成 Token 和 Token ID

02_next_token_prediction.py
    观察固定 prompt 后的 Top-k 下一个 Token

03_token_level_scoring.py
    观察完整文本中每个目标 Token 的概率

04_candidate_sequence_scoring.py
    正确计算多 Token 候选序列的联合概率

05_compare_texts.py
    用 total log probability、NLL、Perplexity 比较文本
```

整个项目的核心是：先明确“当前到底要预测哪个 Token”，再使用正确的 logits 位置和目标 Token ID 取概率。
