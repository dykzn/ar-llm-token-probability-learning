import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "gpt2"


def score_text(text: str, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        logits = model(**inputs).logits

    prediction_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = torch.log_softmax(prediction_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    n_scored = token_log_probs.shape[1]
    total_log_probability = token_log_probs.sum().item()
    nll = -token_log_probs.mean().item()

    return {
        "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        "ids": input_ids[0].tolist(),
        "n_scored": n_scored,
        "total_log_probability": total_log_probability,
        "nll": nll,
        "perplexity": math.exp(nll),
    }


def main() -> None:
    texts = [
        "The capital of France is Paris.",
        "The capital of France is London.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()

    for text in texts:
        result = score_text(text, tokenizer, model)
        print("text:", repr(text))
        print("tokens:", result["tokens"])
        print("ids:", result["ids"])
        print("scored tokens:", result["n_scored"])
        print(
            "total_log_probability:",
            f"{result['total_log_probability']:.10f}",
        )
        print("nll:", f"{result['nll']:.10f}")
        print("perplexity:", f"{result['perplexity']:.10f}")
        print("-" * 60)


if __name__ == "__main__":
    main()

