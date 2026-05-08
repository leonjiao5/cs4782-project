#!/usr/bin/env python3
"""
Display AMC12 checkpoint sweep results as step-vs-accuracy tables.

Usage:
    python code/compare_sweep.py                     # all runs
    python code/compare_sweep.py --run lora_train_light
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from code.config import RESULTS_DIR

SWEEP_DIR = os.path.join(RESULTS_DIR, "tables", "sweep")

BENCHMARKS = ["amc122024", "amc122025"]
MODES = ["greedy", "maj16"]
BENCHMARK_LABELS = {
    "amc122024": "AMC12-2024",
    "amc122025": "AMC12-2025",
}


def load_sweep_results(run_filter=None):
    """Returns {run_name: {step: {benchmark: {mode: accuracy}}}}"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    pattern = os.path.join(SWEEP_DIR, "*.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        m = re.match(r"^(.+)_ckpt(\d+)_([^_]+)_(greedy|maj16)\.json$", fname)
        if not m:
            continue
        run, step, bench, mode = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if run_filter and run != run_filter:
            continue
        if bench not in BENCHMARKS:
            continue
        with open(path) as f:
            data = json.load(f)
        acc = data.get("accuracy", 0.0)
        pass_k = data.get("pass@16")
        results[run][step][bench][mode] = {"accuracy": acc, "pass@16": pass_k}
    return results


def fmt(val):
    if val is None:
        return "  —   "
    return f"{val*100:5.1f}%"


def display(results):
    if not results:
        print(f"No sweep results found in {SWEEP_DIR}")
        print("Submit eval_amc12_sweep.sub jobs first.")
        return

    col_headers = []
    for b in BENCHMARKS:
        col_headers += [f"{BENCHMARK_LABELS[b]} greedy", f"{BENCHMARK_LABELS[b]} maj@16"]

    for run in sorted(results.keys()):
        steps_data = results[run]
        steps = sorted(steps_data.keys())

        print(f"\n{'═'*80}")
        print(f"  {run}")
        print(f"{'─'*80}")

        header = f"  {'Step':>6}  |"
        for h in col_headers:
            header += f"  {h:>17}  |"
        print(header)
        print(f"  {'─'*6}--" + ("--------------------" * len(col_headers)))

        best = {(b, m): (None, -1) for b in BENCHMARKS for m in MODES}

        for step in steps:
            row = f"  {step:>6}  |"
            for bench in BENCHMARKS:
                for mode in MODES:
                    entry = steps_data[step].get(bench, {}).get(mode)
                    if entry is None:
                        row += f"  {'pending':>17}  |"
                        continue
                    key = "accuracy" if mode == "greedy" else "pass@16"
                    val = entry.get(key)
                    row += f"  {fmt(val):>17}  |"
                    if val is not None and val > best[(bench, mode)][1]:
                        best[(bench, mode)] = (step, val)
            print(row)

        # Best checkpoint row
        best_row = f"  {'BEST':>6}  |"
        for bench in BENCHMARKS:
            for mode in MODES:
                bstep, bval = best[(bench, mode)]
                if bstep is None:
                    best_row += f"  {'—':>17}  |"
                else:
                    best_row += f"  {f'ckpt-{bstep} ({fmt(bval)})':>17}  |"
        print(f"  {'─'*6}--" + ("--------------------" * len(col_headers)))
        print(best_row)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Filter to single run name")
    args = parser.parse_args()

    results = load_sweep_results(run_filter=args.run)
    display(results)


if __name__ == "__main__":
    main()
