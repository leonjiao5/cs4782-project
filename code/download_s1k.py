"""
Download s1K-1.1 from HuggingFace and convert to our training JSONL format.

Usage:
    python code/download_s1k.py

Output: data/train_s1k/train_s1k.jsonl
Each row: {"problem": ..., "solution": <thinking_trace + answer>, "answer": "", "source": ...}
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datasets import load_dataset
from code.config import DATA_DIR


def main():
    print("Downloading simplescaling/s1K-1.1 from HuggingFace...")
    ds = load_dataset(
        "simplescaling/s1K-1.1",
        split="train",
        cache_dir=os.path.join(os.path.dirname(DATA_DIR), "hf_cache"),
    )
    print(f"Loaded {len(ds)} examples. Columns: {ds.column_names}")

    out_dir = os.path.join(DATA_DIR, "train_s1k")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train_s1k.jsonl")

    written = skipped = 0
    with open(out_path, "w") as f:
        for row in ds:
            # Prefer DeepSeek R1 traces (higher quality); fall back to Gemini
            if row.get("deepseek_thinking_trajectory") and row.get("deepseek_attempt"):
                thinking = row["deepseek_thinking_trajectory"]
                attempt  = row["deepseek_attempt"]
            elif row.get("gemini_thinking_trajectory") and row.get("gemini_attempt"):
                thinking = row["gemini_thinking_trajectory"]
                attempt  = row["gemini_attempt"]
            else:
                skipped += 1
                continue

            thinking = (thinking or "").strip()
            attempt  = (attempt  or "").strip()
            if not thinking or not attempt:
                skipped += 1
                continue

            # Concat thinking trace + final answer attempt as the assistant "solution"
            solution = thinking + "\n\n" + attempt

            record = {
                "problem": row["question"],
                "solution": solution,
                "answer": "",
                "source": row.get("source_type", "s1k"),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} examples to {out_path}")
    if skipped:
        print(f"Skipped {skipped} examples (missing reasoning traces)")

    # Quick stats
    rows = []
    with open(out_path) as f:
        for line in f:
            rows.append(json.loads(line))
    avg_sol_chars = sum(len(r["solution"]) for r in rows) / len(rows)
    avg_prob_chars = sum(len(r["problem"]) for r in rows) / len(rows)
    print(f"Avg problem length: {avg_prob_chars:.0f} chars")
    print(f"Avg solution length: {avg_sol_chars:.0f} chars (~{avg_sol_chars/4:.0f} tokens)")


if __name__ == "__main__":
    main()
