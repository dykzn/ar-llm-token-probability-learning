# 模型文件说明

本仓库故意不上传语言模型权重。原因是 `models/gpt2/model.safetensors` 大约 523 MB，普通 GitHub 单文件上传有大小限制；而且权重不是学习代码和文档本身。

仓库中保留了运行 tokenizer 所需的文件，以及 GPT-2 的配置文件。要运行 `experiments/02` 到 `experiments/05`，请在本地下载 `model.safetensors`。

## 使用 `hfd.sh` 下载权重

在项目根目录执行。下面的命令使用直连 Hugging Face，不使用代理：

```bash
HF_ENDPOINT=https://huggingface.co bash tools/hfd.sh \
  openai-community/gpt2 \
  --include model.safetensors \
  --local-dir models/gpt2 \
  --no-proxy \
  --tool wget
```

下载完成后，确认下面的文件存在：

```text
models/gpt2/model.safetensors
```

不要把该文件提交到 Git。项目根目录的 `.gitignore` 已经将模型权重排除。
