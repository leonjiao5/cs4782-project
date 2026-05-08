#!/usr/bin/env python3
"""
Generate publication-ready figures and results table from AMC12/MATH eval results.

Usage:
  python code/plot_figures.py
  python code/plot_figures.py --out-dir results/figures
"""
from __future__ import annotations

import argparse
import csv
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
import matplotlib.patches as mpatches
import numpy as np

from code.config import RESULTS_DIR

SWEEP_DIR   = Path(RESULTS_DIR) / "tables" / "sweep"
TABLES_DIR  = Path(RESULTS_DIR) / "tables"
MATH_DIR    = Path(RESULTS_DIR) / "tables" / "math"
FIGURES_DIR = Path(_ROOT) / "results" / "figures"

FAMILY_COLOR = {
    "lora":     "#4C72B0",
    "dora":     "#DD8452",   # peft-DoRA = "DoRA"
    "baseline": "#8C8C8C",
}
SCORING_COLOR = {"logit": "#2196F3", "twopass": "#FF9800", "greedy": "#F44336"}

matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.titlesize"] = 12

MISSING: list[str] = []

# ── Data loaders ──────────────────────────────────────────────────────────────

_SWEEP_RE = re.compile(
    r"^(?P<run_id>.+?)_(?:ckpt(?P<ckpt>\d+)|step(?P<step>\d+)|(?P<avgall>avgall))"
    r"_(?P<bm>amc122024|amc122025|aime2024|aime2025)"
    r"_(?P<mode>greedy|logit|twopass|maj16|strong_greedy|pass4)$",
    re.IGNORECASE,
)

def _load_acc(path: Path) -> float | None:
    try:
        d = json.load(open(path))
        acc = d.get("accuracy")
        if acc is None:
            nc, nt = d.get("n_correct"), d.get("n_total")
            if nc is not None and nt:
                acc = nc / nt
        if acc is None and "results" in d:
            corr = sum(1 for r in d["results"] if r.get("correct"))
            tot  = len(d["results"])
            acc  = corr / tot if tot else None
        return round(float(acc) * 100, 1) if acc is not None else None
    except Exception:
        return None


def load_sweep(sweep_dir: Path = SWEEP_DIR) -> list[dict]:
    records = []
    for path in sorted(sweep_dir.glob("*.json")):
        if path.name.endswith(".partial"):
            continue
        m = _SWEEP_RE.match(path.stem)
        if not m:
            continue
        step = int(m.group("ckpt") or m.group("step")) if (m.group("ckpt") or m.group("step")) else None
        tag  = "avgall" if m.group("avgall") else None
        acc  = _load_acc(path)
        if acc is None:
            continue
        records.append({
            "run_id": m.group("run_id"),
            "step": step, "tag": tag,
            "bm": m.group("bm").lower(),
            "mode": m.group("mode").lower(),
            "acc": acc, "path": str(path),
        })
    return records


_FLAT_RE = re.compile(
    r"^(?P<run_id>.+)_(?P<bm>amc122024|amc122025|aime2024|aime2025)"
    r"_(?P<mode>greedy|logit|twopass|maj16|strong_greedy)$",
    re.IGNORECASE,
)

def load_flat(tables_dir: Path = TABLES_DIR) -> list[dict]:
    records = []
    for path in sorted(tables_dir.glob("*.json")):
        if path.name.endswith(".partial"):
            continue
        m = _FLAT_RE.match(path.stem)
        if not m:
            continue
        acc = _load_acc(path)
        if acc is None:
            continue
        records.append({
            "run_id": m.group("run_id"),
            "step": None, "tag": "flat",
            "bm": m.group("bm").lower(),
            "mode": m.group("mode").lower(),
            "acc": acc, "path": str(path),
        })
    return records


def load_math(math_dir: Path = MATH_DIR) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(math_dir.glob("*.json")):
        if path.name.endswith(".partial"):
            continue
        try:
            d = json.load(open(path))
            acc = d.get("accuracy")
            if acc is None:
                continue
            run_id = path.stem.replace("_math_greedy", "")
            out[run_id] = {
                "overall": round(float(acc) * 100, 1),
                "by_level": {k: round(float(v) * 100, 1)
                             for k, v in (d.get("by_level") or {}).items()},
            }
        except Exception:
            continue
    return out


def best(records: list[dict], run_id: str, bm: str, mode: str) -> float | None:
    vals = [r["acc"] for r in records
            if r["run_id"] == run_id and r["bm"] == bm and r["mode"] == mode
            and r["acc"] is not None]
    return max(vals) if vals else None


def series(records: list[dict], run_id: str, bm: str, mode: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in records:
        if r["run_id"] == run_id and r["bm"] == bm and r["mode"] == mode \
                and r["step"] is not None and r["acc"] is not None:
            out[r["step"]] = max(out.get(r["step"], 0), r["acc"])
    return dict(sorted(out.items()))


def _miss(label: str) -> None:
    MISSING.append(label)


# ── Shared bar-label helper ───────────────────────────────────────────────────

def _label_bar(ax, bar, val, fontsize=8.5, offset=0.8, color=None):
    if val and val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
                color=color or "black")


# ── Figure A: Three scoring methods ──────────────────────────────────────────

def fig_a_scoring_methods(sweep: list[dict], flat: list[dict], out: Path) -> None:
    all_rec = sweep + flat
    BM = "amc122024"

    # (label, run_id, family, no_twopass)
    # Ordered: Baseline → LoRA → DoRA (eye lands on DoRA last)
    groups = [
        ("Baseline\n(Instruct)",   "baseline_instruct",              "baseline", False),
        # ("LoRA\nlight (base)",     "lora_train_light",               "lora",     False),
        # ("LoRA\nlight (inst)",     "lora_train_light_instruct",      "lora",     False),
        # ("LoRA\nheavy",            "lora_train_heavy",               "lora",     False),
        ("DoRA\nlight (base)",     "peft_dora_train_light",          "dora",     False),
        ("DoRA\nlight (inst)",     "peft_dora_train_light_instruct", "dora",     False),
#        ("DoRA\nr4 (base)",        "peft_dora_train_light_r4",       "dora",     False),
        ("DoRA\nheavy (base)",     "peft_dora_train_heavy",          "dora",     True),
    ]

    x  = np.arange(len(groups))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(8, 5.5))

    logit_v   = [best(all_rec, g[1], BM, "logit")   for g in groups]
    twopass_v = [best(all_rec, g[1], BM, "twopass") for g in groups]
    greedy_v  = [best(all_rec, g[1], BM, "greedy")  for g in groups]

    # Shade DoRA region behind everything
    dora_idx = [i for i, g in enumerate(groups) if g[2] == "dora"]
    if dora_idx:
        ax.axvspan(dora_idx[0] - 0.5, dora_idx[-1] + 0.5,
                   alpha=0.07, color=FAMILY_COLOR["dora"], zorder=0)
        ax.text(np.mean(dora_idx), 60.5, "DoRA models",
                ha="center", fontsize=9.5, color=FAMILY_COLOR["dora"],
                style="italic", fontweight="bold")

    def _bar(pos, vals, color, label, hatch=""):
        bars = ax.bar(pos, [v or 0 for v in vals], w,
                      color=color, alpha=0.85, label=label,
                      edgecolor="white", hatch=hatch)
        for bar, v in zip(bars, vals):
            _label_bar(ax, bar, v)
        return bars

    b_logit   = _bar(x - w, logit_v,   SCORING_COLOR["logit"],   "Logit")
    b_twopass = _bar(x,     twopass_v, SCORING_COLOR["twopass"], "Two-pass", hatch="//")
    b_greedy  = _bar(x + w, greedy_v,  SCORING_COLOR["greedy"],  "Greedy",   hatch="xx")

    # Dim non-DoRA bars
    for i, g in enumerate(groups):
        if g[2] != "dora":
            for container in (b_logit, b_twopass, b_greedy):
                container[i].set_alpha(0.4)

    # Mark groups where twopass was not evaluated
    for i, g in enumerate(groups):
        if g[3]:
            ax.text(x[i], 1, "†", ha="center", va="bottom", fontsize=11,
                    color=SCORING_COLOR["twopass"], fontweight="bold")

    for i, (lv, gv) in enumerate(zip(logit_v, greedy_v)):
        if lv is not None and gv is not None:
            gap = lv - gv
            peak = max(v or 0 for v in [lv, gv, twopass_v[i]])
            col = "#2e7d32" if abs(gap) < 10 else "#b71c1c"
            ax.text(x[i], peak + 4,
                    f"Δ{gap:+.0f}pp", ha="center", fontsize=8.5,
                    color=col, fontweight="bold")

    bl_logit = _load_acc(TABLES_DIR / "baseline_amc122024_logit.json") or 32.0
    ax.axhline(bl_logit, color="#555", linestyle="--", linewidth=1.2,
               label=f"Baseline (base) logit = {bl_logit:.0f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("AMC12 2024 Accuracy (%)")
    ax.set_ylim(0, 64)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("DoRA Knows, but Can’t Say",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figA_scoring_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figA_scoring_methods.png")


# ── Figure B: Gap vs dataset ──────────────────────────────────────────────────

def fig_b_gap_vs_dataset(sweep: list[dict], flat: list[dict], out: Path) -> None:
    all_rec = sweep + flat
    BM = "amc122024"

    # DoRA entries first so they appear at top of horizontal bar chart
    entries = [
        ("DoRA light (inst)",  "peft_dora_train_light_instruct", "dora"),
        ("DoRA light (base)",  "peft_dora_train_light",          "dora"),
#        ("DoRA r2 (base)",     "peft_dora_train_light_r2",       "dora"),
#        ("DoRA r4 (base)",     "peft_dora_train_light_r4",       "dora"),
        ("DoRA heavy (base)",  "peft_dora_train_heavy",          "dora"),
        ("LoRA light (inst)",  "lora_train_light_instruct",      "lora"),
        ("LoRA light (base)",  "lora_train_light",               "lora"),
        ("LoRA heavy (base)",  "lora_train_heavy",               "lora"),
        ("Baseline (instruct)","baseline_instruct",              "baseline"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y_pos = np.arange(len(entries))

    # Shade DoRA rows
    dora_rows = [i for i, e in enumerate(entries) if e[2] == "dora"]
    if dora_rows:
        ax.axhspan(dora_rows[-1] - 0.5, dora_rows[0] + 0.5,
                   alpha=0.07, color=FAMILY_COLOR["dora"], zorder=0)

    for i, (label, rid, family) in enumerate(entries):
        lv = best(all_rec, rid, BM, "logit")
        gv = best(all_rec, rid, BM, "greedy")
        color = FAMILY_COLOR[family]
        alpha_dim = 1.0 if family == "dora" else 0.55

        if lv is not None:
            ax.barh(i, lv, color=color, alpha=0.35 * alpha_dim, height=0.5)
            ax.text(lv + 0.4, i, f"{lv:.0f}%", va="center", fontsize=9,
                    color=color, alpha=alpha_dim)
        if gv is not None:
            ax.barh(i, gv, color=color, alpha=0.9 * alpha_dim, height=0.5)
            ax.text(gv + 0.4, i, f"{gv:.0f}%", va="center", fontsize=9,
                    fontweight="bold", alpha=alpha_dim)

        if lv is not None and gv is not None:
            gap = lv - gv
            mid = (lv + gv) / 2
            ax.annotate("", xy=(lv, i + 0.28), xytext=(gv, i + 0.28),
                        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.3,
                                        alpha=alpha_dim))
            col = "#b71c1c" if gap > 15 else "#2e7d32"
            ax.text(mid, i + 0.44, f"Δ{gap:.0f}pp",
                    ha="center", fontsize=8.5, color=col, alpha=alpha_dim)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in entries], fontsize=10)
    ax.set_xlabel("AMC12 2024 Accuracy (%)")
    ax.set_xlim(0, 58)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    legend_patches = [
        mpatches.Patch(color="#555", alpha=0.9, label="Greedy (solid)"),
        mpatches.Patch(color="#555", alpha=0.35, label="Logit (faded)"),
    ] + [mpatches.Patch(color=FAMILY_COLOR[f], label=f.capitalize())
         for f in ["lora", "dora", "baseline"]]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.set_title("Knowledge-Expression Gap decreases with increased training data\n"
                 ,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figB_gap_vs_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figB_gap_vs_dataset.png")


# ── Figure C: Checkpoint progression ─────────────────────────────────────────

def fig_c_progression(sweep: list[dict], out: Path) -> None:
    # DoRA panels first (top row), LoRA for comparison (bottom row)
    runs = [
        ("peft_dora_train_light",        "DoRA light (base)",      "dora"),
        ("peft_dora_train_light_r4",     "DoRA r4 (base)",         "dora"),
        ("peft_dora_train_light_instruct","DoRA light (instruct)", "dora"),
        ("lora_train_light",             "LoRA light (base)",      "lora"),
        ("lora_train_light_instruct",    "LoRA light (instruct)",  "lora"),
        ("lora_train_heavy",             "LoRA heavy (base)",      "lora"),
    ]

    BASELINE = 32.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, (rid, title, family) in zip(axes.flatten(), runs):
        color = FAMILY_COLOR[family]
        s24 = series(sweep, rid, "amc122024", "logit")
        s25 = series(sweep, rid, "amc122025", "logit")

        if s24:
            steps = sorted(s24)
            ax.plot(steps, [s24[s] for s in steps], "o-", color=color,
                    lw=2.2, ms=7, label="AMC2024")
        if s25:
            steps25 = sorted(s25)
            ax.plot(steps25, [s25[s] for s in steps25], "s--", color=color,
                    lw=1.8, ms=6, alpha=0.65, label="AMC2025")

        ax.axhline(BASELINE, color="#888", linestyle=":", lw=1.2,
                   label=f"Baseline logit ({BASELINE:.0f}%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Logit accuracy (%)")
        ax.set_ylim(0, 52)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("DoRA Training Dynamics: Logit Accuracy Across Checkpoints\n"
                 "DoRA r4 still improving at step 700 (not saturated); "
                 "DoRA light peaks then declines — LoRA shown for comparison",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figC_checkpoint_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figC_checkpoint_progression.png")


# ── Figure D: Rank ablation ───────────────────────────────────────────────────

def fig_d_rank_ablation(sweep: list[dict], out: Path) -> None:
    BM = "amc122024"
    families = [
        ("LoRA",  "lora_train_light",       "lora"),
        ("DoRA",  "peft_dora_train_light",  "dora"),
    ]
    ranks = [2, 4, 8]

    def run_name(base_prefix, rank, instruct):
        sfx = "_instruct" if (instruct and rank == 8) else ("_inst" if instruct else "")
        if rank == 8:
            return base_prefix + ("_instruct" if instruct else "")
        return f"{base_prefix}_r{rank}" + ("_inst" if instruct else "")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, (family_label, base_prefix, family_key) in zip(axes, families):
        color = FAMILY_COLOR[family_key]
        x = np.arange(len(ranks))
        w = 0.35

        base_vals = [best(sweep, run_name(base_prefix, r, False), BM, "logit") for r in ranks]
        inst_vals = [best(sweep, run_name(base_prefix, r, True),  BM, "logit") for r in ranks]

        for vals, offset, hatch, label in [
            (base_vals, -w/2, "",    "Base model"),
            (inst_vals,  w/2, "//", "Instruct model"),
        ]:
            for j, (rank, v) in enumerate(zip(ranks, vals)):
                if v is not None:
                    bar = ax.bar(x[j] + offset, v, w, color=color, alpha=0.85,
                                 hatch=hatch, edgecolor="white",
                                 label=label if j == 0 else "")
                    ax.text(x[j] + offset, v + 0.5, f"{v:.0f}%",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
                else:
                    ax.bar(x[j] + offset, 4, w, color="#ddd", edgecolor="#aaa",
                           hatch="///", alpha=0.4,
                           label="Missing" if j == 0 else "")
                    _miss(f"Fig D: {run_name(base_prefix, rank, hatch != '')} logit")

        ax.set_xticks(x)
        ax.set_xticklabels([f"r={r}" for r in ranks])
        ax.set_title(family_label, fontweight="bold")
        ax.set_ylim(0, 52)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Peak logit accuracy % (AMC2024)")
        ax.legend(fontsize=9)

    fig.suptitle("DoRA: Logit Score is Rank-Invariant — Knowledge is Retained Regardless of Rank\n"
                 "r2 ≈ r4 ≈ r8 (~42–44% logit) for DoRA; LoRA shown for comparison",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figD_rank_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figD_rank_ablation.png")


# ── Figure E: Greedy degradation in instruct models ──────────────────────────

def fig_e_greedy_degradation(sweep: list[dict], flat: list[dict], out: Path) -> None:
    all_rec = sweep + flat
    BM = "amc122024"

    runs = [
        ("lora_train_light_instruct",  "LoRA r8 (instruct)", "#1565C0"),
        ("lora_train_light_r4_inst",   "LoRA r4 (instruct)", "#0D47A1"),
        ("lora_train_light_r2_inst",   "LoRA r2 (instruct)", "#42A5F5"),
    ]

    fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(13, 5))

    for rid, label, color in runs:
        g_s = series(all_rec, rid, BM, "greedy")
        l_s = series(all_rec, rid, BM, "logit")
        if g_s:
            steps = sorted(g_s)
            ax_g.plot(steps, [g_s[s] for s in steps], "o-",
                      color=color, lw=2.2, ms=8, label=label)
            for s in steps:
                ax_g.annotate(f"{g_s[s]:.0f}%", (s, g_s[s]),
                              textcoords="offset points", xytext=(0, 8),
                              ha="center", fontsize=8)
        if l_s:
            steps = sorted(l_s)
            ax_l.plot(steps, [l_s[s] for s in steps], "s-",
                      color=color, lw=2.2, ms=8, label=label)

    bl_g = best(all_rec, "baseline_instruct", BM, "greedy")
    if bl_g:
        for ax in (ax_g, ax_l):
            ax.axhline(bl_g, color="#888", linestyle="--", lw=1.2,
                       label=f"Baseline instruct greedy ({bl_g:.0f}%)")

    for ax, title, ylabel in [
        (ax_g, "Greedy Accuracy vs Training Step", "Greedy accuracy (%)"),
        (ax_l, "Logit Accuracy vs Training Step",  "Logit accuracy (%)"),
    ]:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 56)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)

    fig.suptitle("Fine-tuning Instruct Models: Generation Collapses While Knowledge Survives\n"
                 "Greedy peaks at step 100 (48%) then falls; logit stays high — "
                 "same knowledge-expression gap seen in DoRA",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figE_greedy_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figE_greedy_degradation.png")


# ── Figure F: MATH benchmark ─────────────────────────────────────────────────

def fig_f_math_benchmark(sweep: list[dict], flat: list[dict],
                          math: dict[str, dict], out: Path) -> None:
    all_rec = sweep + flat

    # (label, amc_run_id, math_run_id, family)
    entries = [
        ("Baseline\n(base)",     "qwen25_3b_base",       "qwen25_3b_base",       "baseline"),
        ("Baseline\n(instruct)", "qwen25_3b_instruct",   "qwen25_3b_instruct",   "baseline"),
        ("DoRA\nlight",          "peft_dora_train_light","peft_dora_train_light", "dora"),
        ("LoRA\nheavy",          "lora_train_heavy",     "lora_train_heavy",      "lora"),
        #("LoRA s1k\n(instruct)", "lora_train_s1k_inst",  "lora_train_s1k_inst",  "lora"),
    ]

    x   = np.arange(len(entries))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))

    amc_vals  = []
    math_vals = []
    colors    = []

    for label, amc_run_id, math_run_id, family in entries:
        mv = math.get(math_run_id, {}).get("overall")
        av = best(all_rec, amc_run_id, "amc122024", "greedy") if amc_run_id else None
        math_vals.append(mv)
        amc_vals.append(av)
        colors.append(FAMILY_COLOR[family])

    bars_amc  = ax.bar(x - w/2, [v or 0 for v in amc_vals], w,
                       color=colors, alpha=0.85, label="AMC2024 greedy",
                       edgecolor="white")
    bars_math = ax.bar(x + w/2, [v or 0 for v in math_vals], w,
                       color=colors, alpha=0.45, label="MATH overall",
                       edgecolor="white", hatch="//")

    for bar, v in zip(bars_amc, amc_vals):
        _label_bar(ax, bar, v)
    for bar, v in zip(bars_math, math_vals):
        _label_bar(ax, bar, v)

    # Reference lines
    bl_math = math.get("qwen25_3b_base", {}).get("overall", 43.8)
    ax.axhline(bl_math, color="#555", linestyle=":", lw=1.3,
               label=f"Baseline (base) MATH = {bl_math:.0f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in entries], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 62)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#555", alpha=0.85, label="AMC2024 greedy (target task)"),
        mpatches.Patch(color="#555", alpha=0.45, hatch="//", label="MATH overall (transfer)"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="upper right")
    ax.set_title("DoRA Fine-tuning Causes the Largest MATH Regression\n"
                 "DoRA light falls furthest below baseline on general MATH; "
                 "LoRA s1k (small targeted data) nearly preserves it",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figF_math_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figF_math_benchmark.png")


# ── Figure G: MATH by difficulty level ───────────────────────────────────────

def fig_g_math_by_level(math: dict[str, dict], out: Path) -> None:
    runs = [
        ("qwen25_3b_base",        "Baseline (base)",      "#8C8C8C", "o-"),
        ("qwen25_3b_instruct",    "Baseline (instruct)",  "#555555", "s-"),
        ("lora_train_s1k_inst",   "LoRA s1k (instruct)",  "#4C72B0", "^-"),
        ("lora_train_heavy",      "LoRA heavy",           "#2196F3", "D--"),
        ("peft_dora_train_light",  "DoRA light",          "#DD8452", "o--"),
    ]
    levels = ["1", "3", "5"]
    level_labels = ["Level 1\n(Easy)", "Level 3\n(Medium)", "Level 5\n(Hard)"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for run_id, label, color, marker in runs:
        data = math.get(run_id, {}).get("by_level", {})
        vals = [data.get(lv) for lv in levels]
        if any(v is not None for v in vals):
            ax.plot(range(len(levels)), [v or 0 for v in vals],
                    marker, color=color, lw=2.2, ms=9, label=label)
            for j, v in enumerate(vals):
                if v is not None:
                    ax.annotate(f"{v:.0f}%", (j, v),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=8.5, color=color)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(level_labels, fontsize=11)
    ax.set_ylabel("MATH Accuracy (%)")
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title('DoRA the Bad Explorer',
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figG_math_by_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figG_math_by_level.png")


# ── Figure H: AMC2024 vs AMC2025 generalization ───────────────────────────────

def fig_h_amc_generalization(sweep: list[dict], flat: list[dict], out: Path) -> None:
    all_rec = sweep + flat

    # DoRA entries first
    entries = [
        ("DoRA light (base)",   "peft_dora_train_light",          "dora"),
        ("DoRA light (inst)",   "peft_dora_train_light_instruct", "dora"),
        ("DoRA r2 (base)",      "peft_dora_train_light_r2",       "dora"),
        ("DoRA r4 (base)",      "peft_dora_train_light_r4",       "dora"),
        ("DoRA heavy (base)",   "peft_dora_train_heavy",          "dora"),
        ("LoRA heavy",          "lora_train_heavy",               "lora"),
        ("LoRA light (base)",   "lora_train_light",               "lora"),
        ("LoRA light (inst)",   "lora_train_light_instruct",      "lora"),
        ("LoRA r2 (base)",      "lora_train_light_r2",            "lora"),
        ("LoRA r4 (base)",      "lora_train_light_r4",            "lora"),
        ("Baseline (instruct)", "baseline_instruct",              "baseline"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(entries))

    for i, (label, rid, family) in enumerate(entries):
        l24 = best(all_rec, rid, "amc122024", "logit")
        l25 = best(all_rec, rid, "amc122025", "logit")
        color = FAMILY_COLOR[family]

        if l24 is not None:
            ax.barh(i, l24, color=color, alpha=0.9, height=0.5)
            ax.text(l24 + 0.4, i, f"{l24:.0f}%", va="center",
                    fontsize=9, fontweight="bold")
        if l25 is not None:
            ax.barh(i, l25, color=color, alpha=0.4, height=0.5)
            ax.text(l25 + 0.4, i + 0.22, f"{l25:.0f}%", va="center",
                    fontsize=8.5, color=color)

        if l24 is not None and l25 is not None:
            gap = l24 - l25
            ax.text(max(l24, l25) + 3.5, i,
                    f"↓{gap:.0f}pp", va="center", fontsize=8.5, color="#666")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in entries], fontsize=10)
    ax.set_xlabel("Logit Accuracy (%)")
    ax.set_xlim(0, 54)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    legend_patches = [
        mpatches.Patch(color="#555", alpha=0.9, label="AMC 2024 (solid)"),
        mpatches.Patch(color="#555", alpha=0.4,  label="AMC 2025 (faded)"),
    ] + [mpatches.Patch(color=FAMILY_COLOR[f], label=f.capitalize())
         for f in ["lora", "dora", "baseline"]]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.set_title("DoRA and LoRA Generalize Similarly to Unseen 2025 Problems\n"
                 "~8–14pp logit drop from AMC 2024→2025 across all models; "
                 "no method generalizes noticeably better",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "figH_amc_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figH_amc_generalization.png")


# ── Figure I: DoRA vs LoRA hero figure ───────────────────────────────────────

def fig_i_dora_vs_lora(sweep: list[dict], flat: list[dict], out: Path) -> None:
    """Clean head-to-head: logit vs twopass vs greedy for matched LoRA/DoRA pairs."""
    all_rec = sweep + flat
    BM = "amc122024"

    # (label, run_id, family, no_twopass)
    pairs = [
        ("Baseline\n(base)",  "baseline",             "baseline", False),
        ("LoRA\nlight",       "lora_train_light",      "lora",     False),
        ("DoRA\nlight",       "peft_dora_train_light",  "dora",    False),
        ("LoRA\nheavy",       "lora_train_heavy",      "lora",     False),
        ("DoRA\nheavy",       "peft_dora_train_heavy",  "dora",    True),
    ]

    x = np.arange(len(pairs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Shade DoRA columns
    for i, p in enumerate(pairs):
        if p[1] == "lora_train_heavy" or p[1] == "peft_dora_train_heavy":
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color=FAMILY_COLOR["dora"], zorder=0)

    logit_vals   = [best(all_rec, p[1], BM, "logit")   for p in pairs]
    twopass_vals = [best(all_rec, p[1], BM, "twopass") for p in pairs]
    greedy_vals  = [best(all_rec, p[1], BM, "greedy")  for p in pairs]
    colors       = [FAMILY_COLOR[p[2]] for p in pairs]

    bars_l = ax.bar(x - w, [v or 0 for v in logit_vals],   w,
                    color=colors, alpha=0.6,  label="Logit",    edgecolor="white")
    bars_t = ax.bar(x,     [v or 0 for v in twopass_vals],  w,
                    color=colors, alpha=0.75, label="Two-pass", edgecolor="white", hatch="//")
    bars_g = ax.bar(x + w, [v or 0 for v in greedy_vals],   w,
                    color=colors, alpha=0.95, label="Greedy",   edgecolor="white", hatch="xx")

    for bar, v in zip(bars_l, logit_vals):
        _label_bar(ax, bar, v)
    for bar, v in zip(bars_t, twopass_vals):
        _label_bar(ax, bar, v)
    for bar, v in zip(bars_g, greedy_vals):
        _label_bar(ax, bar, v)

    # † where twopass not run
    for i, p in enumerate(pairs):
        if p[3]:
            ax.text(x[i], 1, "†", ha="center", va="bottom", fontsize=11,
                    color=SCORING_COLOR["twopass"], fontweight="bold")

    # Logit–greedy gap annotation
    for i, (lv, gv) in enumerate(zip(logit_vals, greedy_vals)):
        if lv is not None and gv is not None:
            gap = lv - gv
            peak = max(v or 0 for v in [lv, gv, twopass_vals[i]])
            col = "#b71c1c" if gap > 15 else "#2e7d32"
            weight = "bold" if pairs[i][2] == "dora" else "normal"
            ax.text(x[i], peak + 5, f"Δ{gap:+.0f}pp", ha="center",
                    fontsize=11 if pairs[i][2] == "dora" else 9,
                    color=col, fontweight=weight)

    # Dividers: baseline | light | heavy
    ax.axvline(0.5, color="#bbb", linestyle="--", lw=1, zorder=1)
    ax.axvline(2.5, color="#bbb", linestyle="--", lw=1, zorder=1)
    ax.text(1.5, 62, "Train: light dataset", ha="center", fontsize=9, color="#666")
    ax.text(3.5, 62, "Train: heavy dataset", ha="center", fontsize=9, color="#666")


    # Base model reference lines (only logit available; greedy/twopass not evaluated)
    bl_logit = _load_acc(TABLES_DIR / "baseline_amc122024_logit.json") or 32.0
    ax.axhline(bl_logit, color=SCORING_COLOR["logit"], linestyle=":",
               lw=1.6, alpha=0.75)

    legend_patches = [
        mpatches.Patch(color="#555", alpha=0.6,  label="Logit (knowledge probe)"),
        mpatches.Patch(facecolor="#555", edgecolor="#FFF", alpha=0.75, hatch="//", label="Two-pass (reason → read logits)"),
        mpatches.Patch(facecolor="#555", edgecolor="#FFF", alpha=0.95, hatch="xx", label="Greedy (generate & parse)"),
        plt.Line2D([0], [0], color=SCORING_COLOR["logit"], linestyle=":", lw=1.6,
                   label=f"Base model logit = {bl_logit:.0f}% (zero-shot)"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=12)
    ax.set_ylabel("AMC12 2024 Accuracy (%)")
    ax.set_ylim(0, 68)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("DoRA Learns to Speak with Data",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "figI_dora_vs_lora.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figI_dora_vs_lora.png")


# ── Figure J: Logit eval explainer ───────────────────────────────────────────

def fig_j_logit_explainer(out: Path) -> None:
    """Conceptual diagram comparing greedy generation vs logit scoring."""
    fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")

    for ax in (ax_g, ax_l):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, w, h, text, fc="#EEF2FF", ec="#4C72B0", fs=9.5,
            bold=False, radius=0.3, color="black"):
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad={radius}",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=color, zorder=4, wrap=True,
                multialignment="center")

    def arrow(ax, x0, y0, x1, y1, color="#333", lw=1.8):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=16), zorder=5)

    # ── Problem prompt (shared concept) ──────────────────────────────────────
    prompt_text = "Problem:\nIf $2x + 3 = 7$, what is $x$?\n\n(A) 1   (B) 2   (C) 3   (D) 4   (E) 5"
    suffix_text = '↳ append: "\\nThe answer is ("'

    # ── LEFT: Greedy generation ───────────────────────────────────────────────
    ax_g.set_title("Standard: Greedy Generation", fontsize=13, fontweight="bold",
                   pad=10, color="#1565C0")

    box(ax_g, 0.5, 7.5, 9, 2.0, prompt_text, fc="#E3F2FD", ec="#1565C0", fs=8.5)
    arrow(ax_g, 5, 7.5, 5, 6.5)
    box(ax_g, 2.0, 5.2, 6, 1.1, "Transformer\n(auto-regressive)", fc="#E8EAF6", ec="#3949AB", bold=True)
    arrow(ax_g, 5, 5.2, 5, 4.2)

    gen_text = ('"Let me solve this step by step.\n'
                'We have 2x + 3 = 7, so 2x = 4,\n'
                'therefore x = 2. The answer is (B)."')
    box(ax_g, 0.3, 2.5, 9.4, 1.6, gen_text, fc="#FFF9C4", ec="#F9A825", fs=8)
    arrow(ax_g, 5, 2.5, 5, 1.6)
    box(ax_g, 2.8, 0.7, 4.4, 0.8, "Parse answer → B", fc="#E8F5E9", ec="#2E7D32", bold=True, fs=9)

    # annotations
    ax_g.text(0.5, 4.85, "⚠ tokens: ~50–200", fontsize=8, color="#B71C1C",
              style="italic")
    ax_g.text(0.5, 0.25, "✗ can fail to parse   ✗ slow   ✗ model may not express knowledge",
              fontsize=7.5, color="#B71C1C")

    # ── RIGHT: Logit scoring ──────────────────────────────────────────────────
    ax_l.set_title("Our Approach: Logit Scoring", fontsize=13, fontweight="bold",
                   pad=10, color="#DD8452")

    box(ax_l, 0.5, 7.5, 9, 2.0, prompt_text + '\n\n' + suffix_text,
        fc="#FFF3E0", ec="#DD8452", fs=8)
    arrow(ax_l, 5, 7.5, 5, 6.5)
    box(ax_l, 2.0, 5.2, 6, 1.1, "Transformer\n(single forward pass)", fc="#FFF3E0",
        ec="#DD8452", bold=True)
    arrow(ax_l, 5, 5.2, 5, 4.3)

    # Logit bar chart (illustrative values)
    letters = ["A", "B", "C", "D", "E"]
    logits  = [1.2, 4.8, 0.3, -0.5, 0.9]   # illustrative, B is highest
    bar_colors = [FAMILY_COLOR["dora"] if l == "B" else "#BDBDBD" for l in letters]
    bar_x = np.linspace(1.2, 8.8, 5)
    bar_w = 0.9
    max_logit = max(logits)
    min_logit = min(logits)
    scale = 3.5 / (max_logit - min_logit)
    baseline_y = 0.8 - min_logit * scale

    for bx, lv, letter, bc in zip(bar_x, logits, letters, bar_colors):
        bar_h = lv * scale
        rect = plt.Rectangle((bx - bar_w/2, baseline_y),
                              bar_w, bar_h, color=bc, alpha=0.85, zorder=3)
        ax_l.add_patch(rect)
        ax_l.text(bx, baseline_y + bar_h + 0.12, f"{lv:+.1f}",
                  ha="center", fontsize=8, fontweight="bold",
                  color="#DD8452" if letter == "B" else "#555")
        ax_l.text(bx, baseline_y - 0.25, letter,
                  ha="center", fontsize=10, fontweight="bold",
                  color="#DD8452" if letter == "B" else "#444")

    # baseline line
    ax_l.axhline(baseline_y, xmin=0.08, xmax=0.92,
                 color="#999", lw=1, linestyle="--", zorder=2)
    ax_l.text(9.6, baseline_y, "0", va="center", fontsize=8, color="#999")
    ax_l.text(5, 4.05, "Logit scores for each answer token", ha="center",
              fontsize=8, color="#555", style="italic")

    # Arrow from bar chart to result
    arrow(ax_l, 5, 0.45, 5, 0.0)
    # but we're at y=0 already; adjust
    ax_l.annotate("argmax  →  Predicted: B", xy=(5, 0.45), xytext=(5, 0.45),
                  fontsize=10, ha="center", va="top",
                  fontweight="bold", color="#2E7D32")

    ax_l.text(0.5, 4.85, "✓ tokens: exactly 1", fontsize=8, color="#2E7D32",
              style="italic")
    ax_l.text(0.3, 0.05,
              "✓ always parses   ✓ ~100× faster   ✓ reads model's internal belief",
              fontsize=7.5, color="#2E7D32")

    # Divider
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.95],
                               transform=fig.transFigure,
                               color="#CCCCCC", lw=1.5))

    fig.suptitle("Logit Scoring: A Direct Window into What the Model Knows",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out / "figJ_logit_explainer.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved figJ_logit_explainer.png")


# ── Results Table CSV ─────────────────────────────────────────────────────────

def save_results_table(sweep: list[dict], flat: list[dict],
                        math: dict[str, dict], out_csv: Path) -> None:
    all_rec = sweep + flat
    t = TABLES_DIR

    def bl(tag, bm, mode):
        f = t / f"baseline{tag}_{bm}_{mode}.json"
        return _load_acc(f)

    rows = [
        # label, method, dataset, rank, base_model, run_id, math_id
        ("Baseline (3B-base)",    "Baseline", "—",          "—",  "3B-base",    None, "qwen25_3b_base"),
        ("Baseline (3B-instruct)","Baseline", "—",          "—",  "3B-Instruct",None, "qwen25_3b_instruct"),
        ("LoRA light (base)",     "LoRA",     "train_light","r8", "3B-base",    "lora_train_light",              None),
        ("LoRA light (instruct)", "LoRA",     "train_light","r8", "3B-Instruct","lora_train_light_instruct",     None),
        ("LoRA heavy (base)",     "LoRA",     "train_heavy","r8", "3B-base",    "lora_train_heavy",              "lora_train_heavy"),
        ("LoRA r2 (base)",        "LoRA",     "train_light","r2", "3B-base",    "lora_train_light_r2",           None),
        ("LoRA r2 (instruct)",    "LoRA",     "train_light","r2", "3B-Instruct","lora_train_light_r2_inst",      None),
        ("LoRA r4 (base)",        "LoRA",     "train_light","r4", "3B-base",    "lora_train_light_r4",           None),
        ("LoRA r4 (instruct)",    "LoRA",     "train_light","r4", "3B-Instruct","lora_train_light_r4_inst",      None),
        ("LoRA s1k (instruct)",   "LoRA",     "train_s1k",  "r32","3B-Instruct","lora_train_s1k_s1k_inst",      "lora_train_s1k_inst"),
        ("DoRA light (base)",     "DoRA",     "train_light","r8", "3B-base",    "peft_dora_train_light",         "peft_dora_train_light"),
        ("DoRA light (instruct)", "DoRA",     "train_light","r8", "3B-Instruct","peft_dora_train_light_instruct",None),
        ("DoRA r2 (base)",        "DoRA",     "train_light","r2", "3B-base",    "peft_dora_train_light_r2",      None),
        ("DoRA r2 (instruct)",    "DoRA",     "train_light","r2", "3B-Instruct","peft_dora_train_light_r2_inst", None),
        ("DoRA r4 (base)",        "DoRA",     "train_light","r4", "3B-base",    "peft_dora_train_light_r4",      None),
        ("DoRA r4 (instruct)",    "DoRA",     "train_light","r4", "3B-Instruct","peft_dora_train_light_r4_inst", None),
    ]

    fields = ["model","method","train_data","rank","base_model",
              "amc2024_greedy","amc2024_maj16","amc2024_logit","amc2024_twopass",
              "amc2025_greedy","amc2025_maj16","amc2025_logit","amc2025_twopass",
              "aime2024_greedy","aime2024_maj16",
              "math_overall","math_level1","math_level3","math_level5"]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, method, dataset, rank, bm_label, run_id, math_id in rows:
            if run_id is None:
                tag = "_instruct" if "Instruct" in bm_label else ""
                row = {
                    "amc2024_greedy":  bl(tag, "amc122024", "greedy"),
                    "amc2024_maj16":   bl(tag, "amc122024", "maj16"),
                    "amc2024_logit":   bl(tag, "amc122024", "logit"),
                    "amc2024_twopass": None,
                    "amc2025_greedy":  bl(tag, "amc122025", "greedy"),
                    "amc2025_maj16":   bl(tag, "amc122025", "maj16"),
                    "amc2025_logit":   bl(tag, "amc122025", "logit"),
                    "amc2025_twopass": None,
                    "aime2024_greedy": bl(tag, "aime2024", "greedy"),
                    "aime2024_maj16":  bl(tag, "aime2024", "maj16"),
                }
            else:
                row = {
                    "amc2024_greedy":  best(all_rec, run_id, "amc122024", "greedy"),
                    "amc2024_maj16":   best(all_rec, run_id, "amc122024", "maj16"),
                    "amc2024_logit":   best(all_rec, run_id, "amc122024", "logit"),
                    "amc2024_twopass": best(all_rec, run_id, "amc122024", "twopass"),
                    "amc2025_greedy":  best(all_rec, run_id, "amc122025", "greedy"),
                    "amc2025_maj16":   best(all_rec, run_id, "amc122025", "maj16"),
                    "amc2025_logit":   best(all_rec, run_id, "amc122025", "logit"),
                    "amc2025_twopass": best(all_rec, run_id, "amc122025", "twopass"),
                    "aime2024_greedy": _load_acc(t / f"{run_id}_aime2024_greedy.json"),
                    "aime2024_maj16":  _load_acc(t / f"{run_id}_aime2024_maj16.json"),
                }
            m = math.get(math_id or "", {})
            row.update({
                "model": label, "method": method,
                "train_data": dataset, "rank": rank, "base_model": bm_label,
                "math_overall": m.get("overall"),
                "math_level1":  m.get("by_level", {}).get("1"),
                "math_level3":  m.get("by_level", {}).get("3"),
                "math_level5":  m.get("by_level", {}).get("5"),
            })
            w.writerow(row)

    print(f"Saved results_table.csv ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate publication figures and results table.")
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    sweep = load_sweep()
    flat  = load_flat()
    math  = load_math()
    all_r = sweep + flat
    print(f"  {len(sweep)} sweep records, {len(flat)} flat records, "
          f"{len(math)} math results")

    fig_a_scoring_methods(sweep, flat, args.out_dir)
    #fig_b_gap_vs_dataset(sweep, flat, args.out_dir)
    #fig_c_progression(sweep, args.out_dir)
    #fig_d_rank_ablation(sweep, args.out_dir)
    #fig_e_greedy_degradation(sweep, flat, args.out_dir)
    #fig_f_math_benchmark(sweep, flat, math, args.out_dir)
    fig_g_math_by_level(math, args.out_dir)
    #fig_h_amc_generalization(sweep, flat, args.out_dir)
    fig_i_dora_vs_lora(sweep, flat, args.out_dir)
    fig_j_logit_explainer(args.out_dir)

    save_results_table(sweep, flat, math, TABLES_DIR / "results_table.csv")

    print(f"\nAll figures written to {args.out_dir.resolve()}")
    if MISSING:
        print(f"\nMissing data ({len(MISSING)} gaps):")
        for m in MISSING:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
