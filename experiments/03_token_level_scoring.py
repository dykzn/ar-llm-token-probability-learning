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
        outputs = model(**inputs)

    # logits[i] predicts input_ids[i + 1].
    prediction_logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = torch.log_softmax(
        prediction_logits,
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    token_probs = token_log_probs.exp()

    rows = []
    for target_position in range(1, input_ids.shape[1]):
        target_index = target_position - 1
        target_id = int(input_ids[0, target_position])
        context = tokenizer.decode(input_ids[0, :target_position])
        token = tokenizer.decode([target_id])

        rows.append(
            {
                "position": target_position,
                "context": context,
                "token": token,
                "token_id": target_id,
                "probability": float(token_probs[0, target_index]),
                "log_probability": float(
                    token_log_probs[0, target_index]
                ),
            }
        )

    total_log_probability = token_log_probs.sum().item()
    nll = -token_log_probs.mean().item()
    perplexity = math.exp(nll)

    return input_ids, rows, {
        "total_log_probability": total_log_probability,
        "nll": nll,
        "perplexity": perplexity,
    }


def main() -> None:
    text = "The capital of France is Paris."

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()

    input_ids, rows, summary = score_text(
        text,
        tokenizer,
        model,
    )

    print("model:", MODEL_DIR)
    print("text:", repr(text))
    print("input tokens:", tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
    print("input IDs:", input_ids[0].tolist())
    print("scoring convention: ignore the first Token; score the remaining T-1 Tokens")
    print()
    print("position\tcontext\ttoken\ttoken_id\tprobability\tlog_probability")

    for row in rows:
        print(
            row["position"],
            repr(row["context"]),
            repr(row["token"]),
            row["token_id"],
            f"{row['probability']:.10f}",
            f"{row['log_probability']:.10f}",
            sep="\t",
        )

    print()
    print("total_log_probability:", f"{summary['total_log_probability']:.10f}")
    print("nll:", f"{summary['nll']:.10f}")
    print("perplexity:", f"{summary['perplexity']:.10f}")


if __name__ == "__main__":
    main()

