import argparse
import json
from pathlib import Path

from .leakage import filter_aime_leakage
from .sources import load_math_lighteval, load_numinamath_cot, load_openmathinstruct2


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TIERS = {
    "train_light": ["math-lighteval"],
    "train":       ["math-lighteval", "numinamath-cot"],
    "train_scale": ["math-lighteval", "numinamath-cot", "openmathinstruct2"],
}


def _load_aime_problems() -> tuple[list[str], list[str]]:
    problems, answers = [], []
    for year in (2024, 2025):
        d = DATA_DIR / f"aime_{year}"
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text())
            problems.append(r["problem"])
            answers.append(r["answer"])
    return problems, answers


def build_tier(
    tier: str,
    *,
    numinamath_sub_sample: int = 20_000,
    omi2_sub_sample: int = 100_000,
    seed: int = 42,
) -> dict:
    sources = TIERS[tier]
    out_dir = DATA_DIR / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if "math-lighteval" in sources:
        print(f"[{tier}] loading math-lighteval ...")
        rows.extend(load_math_lighteval())
    if "numinamath-cot" in sources:
        print(f"[{tier}] loading numinamath-cot (sub_sample={numinamath_sub_sample}) ...")
        rows.extend(load_numinamath_cot(sub_sample=numinamath_sub_sample, seed=seed))
    if "openmathinstruct2" in sources:
        print(f"[{tier}] loading openmathinstruct2 (sub_sample={omi2_sub_sample}) ...")
        rows.extend(load_openmathinstruct2(sub_sample=omi2_sub_sample, seed=seed))

    print(f"[{tier}] {len(rows)} rows pre-leakage-filter")

    needs_leakage = any(s != "math-lighteval" for s in sources)
    if needs_leakage:
        aime_problems, aime_answers = _load_aime_problems()
        log_path = log_dir / "leakage_drops.jsonl"
        rows, report = filter_aime_leakage(
            rows, aime_problems, aime_answers=aime_answers, log_path=log_path
        )
        print(f"[{tier}] leakage drops: {report}")
    else:
        report = {
            "skipped": True,
            "reason": "MATH-lighteval pre-dates AIME 2024/2025; no leakage filter needed",
        }

    out_jsonl = out_dir / f"{tier}.jsonl"
    with open(out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = out_dir / "leakage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{tier}] wrote {len(rows)} rows -> {out_jsonl}")
    print(f"[{tier}] report -> {report_path}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=list(TIERS), required=True)
    p.add_argument("--numinamath-sub-sample", type=int, default=20_000)
    p.add_argument("--omi2-sub-sample", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_tier(
        args.tier,
        numinamath_sub_sample=args.numinamath_sub_sample,
        omi2_sub_sample=args.omi2_sub_sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
