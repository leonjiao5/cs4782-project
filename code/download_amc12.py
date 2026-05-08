"""
Download AMC12 2024 and 2025 problems from HuggingFace and save as individual JSON files.

Usage:
    python code/download_amc12.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datasets import load_dataset
from code.config import DATA_DIR


def main():
    print("Downloading edev2000/amc12-full from HuggingFace...")
    ds = load_dataset("edev2000/amc12-full", split="train",
                      cache_dir=os.path.join(os.path.dirname(DATA_DIR), "hf_cache"))

    counts = {2024: 0, 2025: 0}
    for row in ds:
        try:
            year = int(row["problem_id"][:4])
        except (ValueError, KeyError):
            continue
        if year not in (2024, 2025):
            continue

        out_dir = os.path.join(DATA_DIR, f"amc12_{year}")
        os.makedirs(out_dir, exist_ok=True)

        pid = row["problem_id"]
        out_path = os.path.join(out_dir, f"{pid}.json")
        problem_text = row.get("question") or row.get("problem") or ""
        with open(out_path, "w") as f:
            json.dump({
                "problem": problem_text,
                "answer": str(row["answer"]).strip().upper(),
                "difficulty": row.get("difficulty"),
            }, f, indent=2)
        counts[year] += 1

    for year, n in counts.items():
        print(f"  amc12_{year}/: {n} problems saved to {os.path.join(DATA_DIR, f'amc12_{year}/')}")

    print("Done.")


if __name__ == "__main__":
    main()
