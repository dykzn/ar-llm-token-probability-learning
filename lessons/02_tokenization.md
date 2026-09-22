# Lesson 02：Tokenization

## Tokenizer 的作用

Tokenizer 把人类文本转换成模型可以处理的 Token ID：

```text
原始文本
→ Token 列表
→ Token ID 列表
→ 输入模型
```

Token 不一定等于完整单词，也可能是单词的一部分、空格加单词，或者标点符号。

## GPT-2 实验观察

```text
Paris  → ['Paris']   → [40313]
 Paris → ['ĠParis']  → [6342]
Paris. → ['Paris', '.'] → [40313, 13]
```

`Ġ` 是 GPT-2 Tokenizer 对前导空格的一种内部表示。

因此，下面两个文本虽然非常相似，但对应不同 Token：

```text
Paris
 Paris
```

这正是作业 Task B 要求检查带空格和不带空格 Token 的原因。

