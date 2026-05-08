#!/usr/bin/env python3
"""
Summarize checkpoint-sweep eval JSONs under results/tables/sweep/.

Filenames follow:
  {run_id}_step{STEP}_{benchmark}_{mode}.json
Example:
  lora_train_light_step100_aime2024_greedy.json
  peft_dora_train_light_instruct_step800_amc122024_maj16.json

For flat leaderboard tables (no step dimension), use:
  python code/compare_results.py

Usage:
  python code/analyze_sweep.py
  python code/analyze_sweep.py --benchmark aime2024
  python code/analyze_sweep.py --run-id lora_train_light --csv sweep_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Allow `python code/analyze_sweep.py` from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from code.config import RESULTS_DIR

SWEEP_DEFAULT = os.path.join(RESULTS_DIR, "tables", "sweep")

# {run_id}_step{N}_{benchmark}_{mode}.json — mode is greedy or maj16
_STEM_RE = re.compile(
    r"^(?P<run_id>.+)_step(?P<step>\d+)_(?P<bm>.+)_(?P<mode>greedy|maj16)\.json$",
    re.IGNORECASE,
)


def parse_sweep_filename(stem: str) -> dict | None:
    m = _STEM_RE.match(stem + ".json")
    if not m:
        return None
    return {
        "run_id": m.group("run_id"),
        "step": int(m.group("step")),
        "benchmark": m.group("bm"),
        "mode": m.group("mode").lower(),
    }


def load_eval_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict:
    out = {
        "accuracy": data.get("accuracy"),
        "n_correct": data.get("n_correct"),
        "n_total": data.get("n_total"),
    }
    for k in (1, 2, 4, 8, 16):
        key = f"pass@{k}"
        if key in data:
            out[key] = data[key]
    return out


def discover_rows(
    sweep_dir: Path,
    run_id_prefix: str | None,
    benchmark: str | None,
    mode: str | None,
) -> list[dict]:
    rows: list[dict] = []
    if not sweep_dir.is_dir():
        return rows

    for path in sorted(sweep_dir.glob("*.json")):
        meta = parse_sweep_filename(path.stem)
        if meta is None:
            print(f"[skip] Unrecognized sweep filename: {path.name}", file=sys.stderr)
            continue
        if run_id_prefix and not meta["run_id"].startswith(run_id_prefix):
            continue
        if benchmark and meta["benchmark"] != benchmark:
            continue
        if mode and meta["mode"] != mode.lower():
            continue

        try:
            raw = load_eval_json(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue

        mets = extract_metrics(raw)
        row = {
            "run_id": meta["run_id"],
            "step": meta["step"],
            "benchmark": meta["benchmark"],
            "mode": meta["mode"],
            "path": str(path),
            **mets,
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "run_id",
        "step",
        "benchmark",
        "mode",
        "accuracy",
        "n_correct",
        "n_total",
        "pass@1",
        "pass@2",
        "pass@4",
        "pass@8",
        "pass@16",
        "path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _pct(x: float | None) -> str:
    if x is None:
        return "   —  "
    return f"{x * 100:5.1f}%"


def print_tables(rows: list[dict]) -> None:
    if not rows:
        print("No matching sweep JSON files.")
        return

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["run_id"], r["benchmark"])
        groups.setdefault(key, []).append(r)

    for (run_id, bm) in sorted(groups.keys()):
        sub = groups[(run_id, bm)]
        by_step: dict[int, dict[str, dict]] = {}
        for r in sub:
            st = r["step"]
            by_step.setdefault(st, {})[r["mode"]] = r

        steps = sorted(by_step.keys())
        print(f"\n{'─' * 72}")
        print(f"  run_id={run_id}   benchmark={bm}")
        print(f"{'─' * 72}")
        print(
            f"  {'step':>6}  {'greedy_acc':>10}  {'maj16_acc':>10}  "
            f"{'pass@8':>8}  {'pass@16':>8}"
        )
        print(f"  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 8}")

        for st in steps:
            g = by_step[st].get("greedy")
            mj = by_step[st].get("maj16")
            ga = g.get("accuracy") if g else None
            ma = mj.get("accuracy") if mj else None
            p8 = mj.get("pass@8") if mj else None
            p16 = mj.get("pass@16") if mj else None
            print(
                f"  {st:6d}  {_pct(ga)}  {_pct(ma)}  {_pct(p8)}  {_pct(p16)}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize results/tables/sweep/*.json by training step."
    )
    ap.add_argument(
        "--sweep-dir",
        default=SWEEP_DEFAULT,
        help=f"Directory with sweep JSONs (default: {SWEEP_DEFAULT})",
    )
    ap.add_argument(
        "--csv",
        metavar="PATH",
        help="Write long-form CSV to PATH",
    )
    ap.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="table: grouped ASCII; csv: print CSV to stdout (use --csv for file)",
    )
    ap.add_argument("--run-id", default=None, help="Filter: run_id prefix")
    ap.add_argument("--benchmark", default=None, help="Filter: benchmark name")
    ap.add_argument("--mode", choices=("greedy", "maj16"), default=None, help="Filter mode")
    ap.add_argument(
        "--also-table",
        action="store_true",
        help="With --csv, also print ASCII tables to stdout",
    )
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = discover_rows(sweep_dir, args.run_id, args.benchmark, args.mode)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote {len(rows)} rows -> {args.csv}")

    if args.format == "csv" and not args.csv:
        w = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "run_id",
                "step",
                "benchmark",
                "mode",
                "accuracy",
                "n_correct",
                "n_total",
                "pass@1",
                "pass@2",
                "pass@4",
                "pass@8",
                "pass@16",
                "path",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    elif args.format == "table" and (not args.csv or args.also_table):
        print_tables(rows)


if __name__ == "__main__":
    main()
