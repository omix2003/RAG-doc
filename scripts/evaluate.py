import argparse
import json
from pathlib import Path

import httpx


def cosine_relevance_score(pred: str, truth: str) -> float:
    pred_tokens = set(pred.lower().split())
    truth_tokens = set(truth.lower().split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    intersection = len(pred_tokens.intersection(truth_tokens))
    return intersection / ((len(pred_tokens) * len(truth_tokens)) ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG relevance benchmark.")
    parser.add_argument("--dataset", required=True, help="JSONL with query/expected fields.")
    parser.add_argument("--api-url", default="http://localhost:8000/ask")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Dataset is empty.")

    scores = []
    with httpx.Client(timeout=60.0) as client:
        for row in rows:
            resp = client.post(
                args.api_url,
                json={"query": row["query"], "session_id": "eval", "history": []},
            )
            resp.raise_for_status()
            answer = resp.json()["answer"]
            scores.append(cosine_relevance_score(answer, row["expected"]))

    avg = sum(scores) / len(scores)
    print(f"Average relevance: {avg:.4f} ({avg * 100:.2f}%)")


if __name__ == "__main__":
    main()
