"""
Download hendrycks/competition_math test split.
Filters to levels 1, 3, 5 and samples 25% per level (seed=42).
Saves to data/math_l135/math_l135.jsonl.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import DATA_DIR
from code.data import extract_boxed_answer

LEVELS = {1, 3, 5}
SAMPLE_FRAC = 0.25
SEED = 42


def main():
    from datasets import load_dataset

    CONFIGS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    by_level: dict[int, list[dict]] = {l: [] for l in LEVELS}
    skipped = 0
    i = 0
    for config in CONFIGS:
        print(f"  Loading {config}...")
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for row in ds:
            level_str = (row.get("level") or "").strip()
            try:
                level = int(level_str.replace("Level", "").strip())
            except ValueError:
                skipped += 1
                i += 1
                continue
            if level not in LEVELS:
                i += 1
                continue
            answer = extract_boxed_answer(row["solution"])
            if answer is None:
                skipped += 1
                i += 1
                continue
            by_level[level].append({
                "id": f"math_L{level}_{i:05d}",
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer,
                "level": level,
                "type": config,
            })
            i += 1

    rng = random.Random(SEED)
    sampled: list[dict] = []
    for level in sorted(LEVELS):
        items = by_level[level]
        k = max(1, round(len(items) * SAMPLE_FRAC))
        selected = rng.sample(items, k)
        selected.sort(key=lambda x: x["id"])
        sampled.extend(selected)
        print(f"  Level {level}: {len(items)} total → {k} sampled")

    out_dir = os.path.join(DATA_DIR, "math_l135")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_l135.jsonl")

    with open(out_path, "w") as f:
        for item in sampled:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(sampled)} problems to {out_path}")
    if skipped:
        print(f"Skipped {skipped} problems (missing level field or no \\boxed{{}} answer)")


if __name__ == "__main__":
    main()
