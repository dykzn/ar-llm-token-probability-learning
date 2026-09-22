# GPT-2 Tokenizer 小实验结果

## 实验目的

观察以下三个文本如何被 GPT-2 Tokenizer 切分：

```text
Paris
 Paris
Paris.
```

重点观察：

- 前导空格是否会影响 Token；
- 句号是否会成为独立 Token；
- Token ID 与 Token 文本的对应关系。

## 实验环境

- 模型 Tokenizer：GPT-2
- Transformers：4.57.1
- Tokenizer 来源：项目内本地目录 `models/gpt2-tokenizer`
- 是否下载完整模型权重：否，只下载了 Tokenizer 文件
- 运行方式：本地离线加载

## 使用代码

```python
import transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "models/gpt2-tokenizer",
    local_files_only=True
)

for text in ["Paris", " Paris", "Paris."]:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print("文本:", repr(text))
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    for token, token_id in zip(tokens, token_ids):
        print(
            "Token =", repr(token),
            "ID =", token_id,
            "还原 =", repr(tokenizer.decode([token_id]))
        )
```

## 实际输出

```text
transformers: 4.57.1
文本: 'Paris'
Tokens: ['Paris']
Token IDs: [40313]
逐个查看:
  Token = 'Paris' ID = 40313 还原 = 'Paris'
----------------------------------------
文本: ' Paris'
Tokens: ['ĠParis']
Token IDs: [6342]
逐个查看:
  Token = 'ĠParis' ID = 6342 还原 = ' Paris'
----------------------------------------
文本: 'Paris.'
Tokens: ['Paris', '.']
Token IDs: [40313, 13]
逐个查看:
  Token = 'Paris' ID = 40313 还原 = 'Paris'
  Token = '.' ID = 13 还原 = '.'
----------------------------------------
```

## 结果解释

### 1. `Paris` 和 ` Paris` 是不同 Token

实验结果为：

```text
Paris  -> Token ID 40313
 Paris -> Token ID 6342
```

虽然它们在人眼看来只差一个空格，但 GPT-2 的词表中有不同的 Token：

```text
'Paris'  -> 'Paris'
' Paris' -> 'ĠParis'
```

其中 `Ġ` 是 GPT-2 Tokenizer 对前导空格的一种内部表示。

### 2. 句号被拆成独立 Token

```text
Paris. -> ['Paris', '.']
```

因此，一个看起来像“单词加标点”的文本，可能实际上包含两个 Token。

### 3. Token ID 是模型词表中的编号

Token ID 只是编号，不是概率，也不是 Token 的长度：

```text
'Paris'  的 Token ID 是 40313
' Paris' 的 Token ID 是 6342
'.'      的 Token ID 是 13
```

这些编号只对 GPT-2 的词表有效。换用其他模型时，Token 切分方式和 ID 都可能变化。

## 与作业的联系

这个实验对应作业中的 Task B。计算候选 Token 概率之前，必须先确认候选文本的 Tokenization。

例如，不能只把 `Paris` 当成一个普通字符串；需要明确分析的是：

```text
模型实际看到的是 'Paris'，还是 ' Paris'？
```

如果一个候选词被拆成多个 Token，后续计算整个候选词的概率时，需要分别考虑这些 Token 的条件概率。
