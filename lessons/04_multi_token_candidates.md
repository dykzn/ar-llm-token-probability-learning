# Lesson 04：多 Token 候选词的完整概率

## 核心方法

如果候选词只有一个 Token，例如：

```text
" Berlin" → ['ĠBerlin']
```

它的概率可以直接从 prompt 最后位置的 probability 向量中读取。

如果候选词有多个 Token，例如：

```text
"Berlin" → ['Ber', 'lin']
```

就需要依次计算：

```text
Ber 在原始 prompt 后面的概率
lin 在 prompt + Ber 后面的概率
```

完整候选词的概率是这些条件概率的乘积；完整 log probability 是它们的和。

## 本次实际结果

```text
" Berlin" → ['ĠBerlin']
完整概率：0.001585415199
```

```text
"Berlin" → ['Ber', 'lin']
Ber 概率：0.0000001242
lin 概率：0.8957989216
完整概率：0.000000111236
```

虽然 `lin` 在 `Ber` 后面的条件概率很高，但 `Ber` 第一步的概率非常低，因此完整的 `Berlin` 概率仍然很低。

## 长度影响

多 Token 候选词进行整体概率比较时，Token 数量越多，连续相乘通常越容易变小。

如果要比较不同长度的候选文本，还可以比较平均 log probability，而不是只比较总概率。

