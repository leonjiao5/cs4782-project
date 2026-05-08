"""
Re-score existing eval result files using the updated cascade is_correct.
Reads all JSON files in results/tables/ that contain a 'responses' field,
re-runs extraction + scoring, writes updated files to results/tables/rescored/.
No re-generation needed — works on saved reasoning traces.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from collections import Counter

from code.utils import is_correct, pass_at_k
from code.data import extract_boxed_answer


def rescore_file(src_path: str, dst_path: str):
    with open(src_path) as f:
        data = json.load(f)

    problems = data.get("problems", [])
    if not problems or "responses" not in problems[0]:
        print(f"  [SKIP] {os.path.basename(src_path)} — no responses field")
        return False

    n_samples = data.get("n_samples", 1)
    n_correct_greedy = 0
    updated_problems = []

    for prob in problems:
        responses = prob["responses"]
        gold = prob["answer"]

        corrects = [is_correct(r, gold) for r in responses]
        pred_answers = [extract_boxed_answer(r) for r in responses]

        if corrects[0]:
            n_correct_greedy += 1

        majority_correct = False
        if n_samples > 1:
            valid = [a for a in pred_answers if a is not None]
            if valid:
                majority_answer = Counter(valid).most_common(1)[0][0]
                majority_correct = majority_answer.strip() == gold.strip()

        updated_problems.append({**prob, "predicted": pred_answers, "correct": corrects, "majority_correct": majority_correct})

    accuracy = n_correct_greedy / len(problems) if problems else 0.0

    pass_k_summary = {}
    if n_samples > 1:
        for k in [1, 2, 4, 8, 16]:
            if k > n_samples:
                break
            per_prob = [pass_at_k(sum(r["correct"]), n_samples, k) for r in updated_problems]
            pass_k_summary[f"pass@{k}"] = sum(per_prob) / len(per_prob)

    updated = {**data, "n_correct": n_correct_greedy, "accuracy": accuracy, **pass_k_summary, "problems": updated_problems}

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(updated, f, indent=2)

    old_acc = data.get("accuracy", 0)
    delta = accuracy - old_acc
    sign = "+" if delta >= 0 else ""
    print(f"  {os.path.basename(src_path)}: {old_acc:.1%} → {accuracy:.1%}  ({sign}{delta:.1%})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", default="results/tables")
    parser.add_argument("--out_dir", default="results/tables/rescored")
    args = parser.parse_args()

    print(f"Re-scoring files in {args.tables_dir} → {args.out_dir}\n")
    changed = improved = 0

    for fname in sorted(os.listdir(args.tables_dir)):
        if not fname.endswith(".json"):
            continue
        src = os.path.join(args.tables_dir, fname)
        dst = os.path.join(args.out_dir, fname)
        ok = rescore_file(src, dst)
        if ok:
            changed += 1
            with open(src) as f:
                old_acc = json.load(f).get("accuracy", 0)
            with open(dst) as f:
                new_acc = json.load(f).get("accuracy", 0)
            if new_acc > old_acc:
                improved += 1

    print(f"\nRescored {changed} files, {improved} improved in accuracy.")


if __name__ == "__main__":
    main()
