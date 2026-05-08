#!/usr/bin/env python3
"""
Aggregate and display results from results/tables/*.json (flat files only).

Checkpoint sweep outputs live under results/tables/sweep/ with step-prefixed names.
Summarize those with:
    python code/analyze_sweep.py

Usage:
    python code/compare_results.py
    python code/compare_results.py --tier train_light
    python code/compare_results.py --benchmark aime2024
"""
import argparse
import json
import os
import glob
from pathlib import Path

from code.config import RESULTS_DIR

TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

METHOD_ORDER = ["baseline", "lora", "peft_dora", "dora"]
BENCHMARK_ORDER = ["aime2024", "aime2025", "amc122024", "amc122025"]
MODE_ORDER = ["greedy", "maj16"]


def load_tables(tier_filter=None, benchmark_filter=None):
    results = {}
    for path in sorted(glob.glob(os.path.join(TABLES_DIR, "*.json"))):
        stem = Path(path).stem  # e.g. lora_train_light_aime2024_greedy
        with open(path) as f:
            data = json.load(f)

        # Parse filename: <method>_<tier>_<benchmark>_<mode>
        # or baseline_<benchmark>_<mode>
        parts = stem.split("_")
        if parts[0] == "baseline":
            method, tier, benchmark, mode = "baseline", "baseline", parts[1], parts[2]
        elif len(parts) >= 4:
            # Could be peft_dora (two words) or lora/dora (one word)
            if parts[0] == "peft":
                method = "peft_dora"
                tier = parts[2]
                benchmark = parts[3]
                mode = parts[4] if len(parts) > 4 else "greedy"
            else:
                method = parts[0]
                tier = parts[1]
                benchmark = parts[2]
                mode = parts[3] if len(parts) > 3 else "greedy"
        else:
            # Legacy filename without explicit mode (old greedy runs)
            method = parts[0]
            tier = parts[1] if len(parts) > 2 else "?"
            benchmark = parts[-1]
            mode = "greedy"

        if tier_filter and tier not in (tier_filter, "baseline"):
            continue
        if benchmark_filter and benchmark != benchmark_filter:
            continue

        key = (method, tier, benchmark, mode)
        results[key] = data

    return results


def fmt_acc(data):
    n, total = data["n_correct"], data["n_total"]
    pct = data["accuracy"] * 100
    return f"{pct:5.1f}% ({n:2d}/{total})"


def fmt_pass(data, k=16):
    val = data.get(f"pass@{k}")
    if val is None:
        return "    —    "
    return f"{val*100:5.1f}%    "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=None, help="Filter by training tier (e.g. train_light)")
    parser.add_argument("--benchmark", default=None, help="Filter by benchmark (aime2024 or aime2025)")
    args = parser.parse_args()

    results = load_tables(tier_filter=args.tier, benchmark_filter=args.benchmark)

    if not results:
        print(f"No results found in {TABLES_DIR}")
        print("Check that eval jobs have completed.")
        return

    # Group by benchmark for display
    benchmarks = sorted({k[2] for k in results}, key=lambda b: BENCHMARK_ORDER.index(b) if b in BENCHMARK_ORDER else 99)

    for benchmark in benchmarks:
        print(f"\n{'─'*65}")
        print(f"  Benchmark: {benchmark.upper()}")
        print(f"{'─'*65}")
        print(f"  {'Method':<18} {'Tier':<14} {'Greedy':>14}  {'Maj@16':>10}  {'pass@4':>8}")
        print(f"  {'─'*18} {'─'*14} {'─'*14}  {'─'*10}  {'─'*8}")

        shown = set()
        for method in METHOD_ORDER:
            for key, data in sorted(results.items()):
                m, tier, bm, mode = key
                if m != method or bm != benchmark:
                    continue
                row_key = (m, tier, bm)
                if row_key in shown:
                    continue
                shown.add(row_key)

                greedy_key = (m, tier, bm, "greedy")
                maj_key    = (m, tier, bm, "maj16")

                greedy_str = fmt_acc(results[greedy_key]) if greedy_key in results else "  pending  "
                maj_str    = fmt_pass(results[maj_key], k=16) if maj_key in results else " pending  "
                p4_str     = fmt_pass(results[maj_key], k=4)  if maj_key in results else "pending "

                label = method if tier == "baseline" else f"{method} ({tier})"
                print(f"  {label:<18} {tier:<14} {greedy_str}  {maj_str}  {p4_str}")

    print()


if __name__ == "__main__":
    main()
