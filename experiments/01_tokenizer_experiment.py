from pathlib import Path

import transformers
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = PROJECT_ROOT / "models" / "gpt2-tokenizer"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True,
    )

    print("transformers:", transformers.__version__)
    print("tokenizer:", TOKENIZER_DIR)
    print()

    for text in ["Paris", " Paris", "Paris."]:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        print("文本:", repr(text))
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)
        print("逐个查看:")

        for token, token_id in zip(tokens, token_ids):
            decoded = tokenizer.decode([token_id])
            print(
                "  Token =",
                repr(token),
                "ID =",
                token_id,
                "还原 =",
                repr(decoded),
            )

        print("-" * 40)


if __name__ == "__main__":
    main()

