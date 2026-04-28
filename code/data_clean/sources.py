import random

from datasets import load_dataset

from code.data import extract_boxed_answer


def load_math_lighteval() -> list[dict]:
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    out = []
    for row in ds:
        ans = extract_boxed_answer(row["solution"])
        if ans is None:
            continue
        out.append({
            "problem": row["problem"],
            "solution": row["solution"],
            "answer": ans,
            "source": "math-lighteval",
        })
    return out


def load_numinamath_cot(
    sub_sample: int | None = None,
    drop_sources: tuple[str, ...] = (),
    seed: int = 42,
) -> list[dict]:
    ds = load_dataset("AI-MO/NuminaMath-CoT", "default", split="train")
    out = []
    for row in ds:
        if row["source"] in drop_sources:
            continue
        ans = extract_boxed_answer(row["solution"])
        if ans is None:
            continue
        out.append({
            "problem": row["problem"],
            "solution": row["solution"],
            "answer": ans,
            "source": "numinamath-cot",
        })
    if sub_sample is not None and len(out) > sub_sample:
        rng = random.Random(seed)
        out = rng.sample(out, sub_sample)
    return out


def load_openmathinstruct2(
    sub_sample: int | None = 100_000,
    seed: int = 42,
) -> list[dict]:
    ds = load_dataset("nvidia/OpenMathInstruct-2", "default", split="train_1M")
    out = []
    for row in ds:
        sol = row["generated_solution"]
        ans = extract_boxed_answer(sol)
        if ans is None:
            continue
        if str(ans).strip() != str(row["expected_answer"]).strip():
            continue
        out.append({
            "problem": row["problem"],
            "solution": sol,
            "answer": ans,
            "source": "openmathinstruct2",
        })
    if sub_sample is not None and len(out) > sub_sample:
        rng = random.Random(seed)
        out = rng.sample(out, sub_sample)
    return out
