"""Merge sharded MATH eval results into a single final JSON."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="e.g. peft_dora_train_heavy")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--math_dir", default="results/tables/math")
    parser.add_argument("--out", default=None, help="Output path (default: <math_dir>/<run_id>_math_greedy.json)")
    args = parser.parse_args()

    math_dir = Path(args.math_dir)
    out_path = Path(args.out) if args.out else math_dir / f"{args.run_id}_math_greedy.json"

    all_problems: list[dict] = []
    meta: dict = {}

    for idx in range(args.num_shards):
        shard_path = math_dir / f"{args.run_id}_math_greedy_shard{idx}of{args.num_shards}.json"
        if not shard_path.exists():
            # Try partial file
            partial = Path(str(shard_path) + ".partial")
            if partial.exists():
                print(f"  WARNING: shard {idx} not complete, using partial: {partial.name}")
                shard_path = partial
            else:
                print(f"  ERROR: shard {idx} missing: {shard_path}", file=sys.stderr)
                sys.exit(1)
        d = json.loads(shard_path.read_text())
        probs = d.get("problems", [])
        all_problems.extend(probs)
        if not meta:
            meta = {k: v for k, v in d.items() if k not in ("problems", "n_correct", "n_total", "accuracy", "by_level", "by_type")}
        print(f"  shard {idx}: {len(probs)} problems")

    # Sort by problem_id for deterministic output
    all_problems.sort(key=lambda r: r["problem_id"])
    n_total = len(all_problems)
    n_correct = sum(1 for r in all_problems if r["correct"][0])
    accuracy = n_correct / n_total if n_total else 0.0

    level_correct: dict = defaultdict(int)
    level_total: dict = defaultdict(int)
    type_correct: dict = defaultdict(int)
    type_total: dict = defaultdict(int)

    for r in all_problems:
        lv = str(r.get("level", "?"))
        tp = r.get("type", "?") or "?"
        level_total[lv] += 1
        type_total[tp] += 1
        if r["correct"][0]:
            level_correct[lv] += 1
            type_correct[tp] += 1

    output = {
        **meta,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
        "by_level": {lv: round(level_correct[lv] / level_total[lv], 4) for lv in sorted(level_total)},
        "by_type":  {tp: round(type_correct[tp]  / type_total[tp],  4) for tp in sorted(type_total)},
        "problems": all_problems,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nMerged {n_total} problems → {out_path}")
    print(f"Overall accuracy: {accuracy:.1%}  ({n_correct}/{n_total})")
    print("By level:", {lv: f"{level_correct[lv]}/{level_total[lv]}" for lv in sorted(level_total)})


if __name__ == "__main__":
    main()
