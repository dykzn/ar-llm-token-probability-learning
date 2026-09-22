from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "gpt2"


def main() -> None:
    text = "The capital of France is"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]
    last_log_probs = torch.log_softmax(last_logits, dim=-1)
    last_probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(last_probs, k=10)

    print("transformers:", transformers.__version__)
    print("torch:", torch.__version__)
    print("model:", MODEL_DIR)
    print("text:", repr(text))
    print("input tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()))
    print("input IDs:", inputs["input_ids"][0].tolist())
    print("logits shape:", tuple(outputs.logits.shape))
    print("last logits shape:", tuple(last_logits.shape))
    print()
    print("Top-10 next-token predictions:")
    print("rank\ttoken\ttoken_id\tprobability\tlog_probability")

    for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), start=1):
        token_id_int = int(token_id)
        token = tokenizer.convert_ids_to_tokens([token_id_int])[0]
        decoded = tokenizer.decode([token_id_int])
        print(
            rank,
            repr(token),
            token_id_int,
            repr(decoded),
            f"{float(prob):.10f}",
            f"{float(last_log_probs[token_id_int]):.10f}",
            sep="\t",
        )

    print()
    print("Candidate first-token probabilities:")
    print("candidate\ttokens\tids\tfirst_probability\tfirst_log_probability")

    for candidate in [" Paris", " London", " Berlin", "Paris", "London", "Berlin"]:
        candidate_ids = tokenizer.encode(
            candidate,
            add_special_tokens=False,
        )
        first_id = candidate_ids[0]
        print(
            repr(candidate),
            repr(tokenizer.convert_ids_to_tokens(candidate_ids)),
            repr(candidate_ids),
            f"{float(last_probs[first_id]):.10f}",
            f"{float(last_log_probs[first_id]):.10f}",
            sep="\t",
        )


if __name__ == "__main__":
    main()

