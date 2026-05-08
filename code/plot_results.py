#!/usr/bin/env python3
"""
Build publication-style figures from results/tables JSON outputs.

Reads:
  - results/tables/*.json (flat eval runs)
  - optionally results/tables/sweep/*.json (checkpoint sweeps)

Writes PNGs under figures/ (project root).

Usage:
  python code/plot_results.py
  python code/plot_results.py --out-dir figures --include-v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.config import RESULTS_DIR
from code.analyze_sweep import SWEEP_DEFAULT, discover_rows

TABLES_DIR = Path(RESULTS_DIR) / "tables"
FIGURES_DEFAULT = Path(_ROOT) / "figures"

# Filename stem: {run_label}_{benchmark}_{mode}
_STEM_RE = re.compile(
    r"^(.+)_(aime2024|aime2025|amc122024|amc122025)_(greedy|maj16)$",
    re.IGNORECASE,
)


def parse_flat_stem(stem: str) -> tuple[str, str, str] | None:
    m = _STEM_RE.match(stem)
    if not m:
        return None
    return m.group(1), m.group(2).lower(), m.group(3).lower()


def load_flat_rows(include_v2: bool) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(TABLES_DIR.glob("*.json")):
        meta = parse_flat_stem(path.stem)
        if meta is None:
            continue
        run_label, benchmark, mode = meta
        if not include_v2 and "_v2" in run_label:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        row = {
            "run_label": run_label,
            "benchmark": benchmark,
            "mode": mode,
            "accuracy": data.get("accuracy"),
            "path": str(path),
        }
        for k in (4, 8, 16):
            key = f"pass@{k}"
            if key in data:
                row[key] = data[key]
        rows.append(row)
    return rows


def collect_series(
    rows: list[dict],
    benchmark: str,
    allowed_labels: list[str],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Returns greedy_acc and pass8 maps keyed by run_label slot (parallel to allowed_labels)."""
    greedy: dict[str, float | None] = {lab: None for lab in allowed_labels}
    p8: dict[str, float | None] = {lab: None for lab in allowed_labels}
    for r in rows:
        if r["benchmark"] != benchmark:
            continue
        lab = r["run_label"]
        if lab not in greedy:
            continue
        if r["mode"] == "greedy" and r["accuracy"] is not None:
            greedy[lab] = r["accuracy"] * 100.0
        if r["mode"] == "maj16":
            v = r.get("pass@8")
            if v is not None:
                p8[lab] = v * 100.0
    return greedy, p8


def plot_grouped_benchmarks(
    out_path: Path,
    title: str,
    benchmarks: list[str],
    labels_order: list[str],
    display_names: list[str],
    rows: list[dict],
) -> None:
    n_bm = len(benchmarks)
    fig, axes = plt.subplots(1, n_bm, figsize=(5 * n_bm, 4.5), squeeze=False)
    x = range(len(labels_order))
    width = 0.35

    for ax_idx, bm in enumerate(benchmarks):
        ax = axes[0, ax_idx]
        greedy, p8 = collect_series(rows, bm, labels_order)
        g_vals = [greedy[lab] if greedy[lab] is not None else 0.0 for lab in labels_order]
        p_vals = [p8[lab] if p8[lab] is not None else 0.0 for lab in labels_order]
        ax.bar([i - width / 2 for i in x], g_vals, width, label="Greedy acc", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], p_vals, width, label="pass@8 (maj16)", color="#DD8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(display_names, rotation=25, ha="right")
        ax.set_ylabel("Percent")
        ax.set_ylim(0, max(100, max(g_vals + p_vals, default=1) * 1.15))
        ax.set_title(bm.upper())
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_step_summary(out_path: Path, sweep_rows: list[dict]) -> None:
    """Bar chart for sweep JSONs: one panel per (run_id, benchmark), steps on x."""
    if not sweep_rows:
        return
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in sweep_rows:
        key = (r["run_id"], r["benchmark"])
        groups.setdefault(key, []).append(r)

    filtered = {k: v for k, v in groups.items() if k[1].startswith("aime")}
    if not filtered:
        filtered = groups

    items = sorted(filtered.items())[:6]
    n = len(items)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n))
    if n == 1:
        axes = [axes]

    for idx, ((run_id, bm), sub) in enumerate(items):
        ax = axes[idx]
        by_step: dict[int, dict[str, dict]] = {}
        for r in sub:
            st = r["step"]
            by_step.setdefault(st, {})[r["mode"]] = r

        steps = sorted(by_step.keys())
        x = range(len(steps))
        width = 0.35
        g_vals = []
        p_vals = []
        for st in steps:
            gr = by_step[st].get("greedy") or {}
            mj = by_step[st].get("maj16") or {}
            ga = gr.get("accuracy")
            p = mj.get("pass@8")
            g_vals.append((ga or 0) * 100)
            p_vals.append((p or 0) * 100 if p is not None else 0.0)

        ax.bar([i - width / 2 for i in x], g_vals, width, label="Greedy acc", color="#55A868")
        ax.bar([i + width / 2 for i in x], p_vals, width, label="pass@8", color="#C44E52")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("Train step")
        ax.set_ylabel("Percent")
        ax.set_title(f"{run_id} — {bm}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Checkpoint sweep (results/tables/sweep)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot figures from results/tables JSON.")
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DEFAULT, help="Output directory for PNGs")
    ap.add_argument("--include-v2", action="store_true", help="Include *_v2_* flat result files")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_flat_rows(include_v2=args.include_v2)

    # Base model (Qwen2.5-3B style) train_light
    base_labels = [
        "baseline",
        "lora_train_light",
        "peft_dora_train_light",
        "dora_train_light",
    ]
    base_display = ["Baseline", "LoRA", "PEFT DoRA", "DoRA"]
    plot_grouped_benchmarks(
        args.out_dir / "aime_train_light_base.png",
        "AIME — train_light (base checkpoint family)",
        ["aime2024", "aime2025"],
        base_labels,
        base_display,
        rows,
    )

    instruct_labels = [
        "baseline_instruct",
        "lora_train_light_instruct",
        "peft_dora_train_light_instruct",
        "dora_train_light_instruct",
    ]
    instruct_display = ["Baseline", "LoRA", "PEFT DoRA", "DoRA"]
    plot_grouped_benchmarks(
        args.out_dir / "aime_train_light_instruct.png",
        "AIME — train_light instruct",
        ["aime2024", "aime2025"],
        instruct_labels,
        instruct_display,
        rows,
    )

    sweep_rows = discover_rows(Path(SWEEP_DEFAULT), None, None, None)
    if sweep_rows:
        plot_sweep_step_summary(args.out_dir / "sweep_checkpoint_curves.png", sweep_rows)

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
