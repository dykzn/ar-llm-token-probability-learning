import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "gpt2"


def score_candidate(prompt: str, candidate: str, tokenizer, model):
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    candidate_ids = tokenizer.encode(
        candidate,
        add_special_tokens=False,
    )

    full_ids = prompt_ids + candidate_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    log_probs = torch.log_softmax(
        outputs.logits[:, :-1, :],
        dim=-1,
    )

    prompt_length = len(prompt_ids)
    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length - 1 + len(candidate_ids),
    )
    target_ids = torch.tensor(candidate_ids, dtype=torch.long)

    selected_log_probs = log_probs[
        0,
        prediction_positions,
        target_ids,
    ]
    selected_probs = selected_log_probs.exp()

    return {
        "candidate": candidate,
        "tokens": tokenizer.convert_ids_to_tokens(candidate_ids),
        "ids": candidate_ids,
        "probabilities": selected_probs.tolist(),
        "log_probabilities": selected_log_probs.tolist(),
        "total_probability": math.exp(selected_log_probs.sum().item()),
        "total_log_probability": selected_log_probs.sum().item(),
    }


def main() -> None:
    prompt = "The capital of France is"
    candidates = [
        " Paris",
        " London",
        " Berlin",
        "Paris",
        "London",
        "Berlin",
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

    print("prompt:", repr(prompt))
    print("prompt tokens:", tokenizer.tokenize(prompt))
    print()

    for candidate in candidates:
        result = score_candidate(
            prompt,
            candidate,
            tokenizer,
            model,
        )

        print("candidate:", repr(result["candidate"]))
        print("tokens:", result["tokens"])
        print("ids:", result["ids"])

        for token, token_id, prob, log_prob in zip(
            result["tokens"],
            result["ids"],
            result["probabilities"],
            result["log_probabilities"],
        ):
            print(
                "  token:", repr(token),
                "id:", token_id,
                "probability:", f"{prob:.10f}",
                "log_probability:", f"{log_prob:.10f}",
            )

        print(
            "total probability:",
            f"{result['total_probability']:.12f}",
        )
        print(
            "total log probability:",
            f"{result['total_log_probability']:.10f}",
        )
        print("-" * 60)


if __name__ == "__main__":
    main()

