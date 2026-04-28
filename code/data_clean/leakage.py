import json
import re
from pathlib import Path


_LATEX_WRAPPERS = re.compile(r"(\$\$|\$|\\\(|\\\)|\\\[|\\\])")
_WHITESPACE = re.compile(r"\s+")


def normalize_problem(s: str) -> str:
    s = s.lower()
    s = _LATEX_WRAPPERS.sub(" ", s)
    s = s.replace("\\", " ")
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


def shingles(s: str, n: int = 5) -> set[str]:
    words = s.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_aime_leakage(
    rows: list[dict],
    aime_problems: list[str],
    aime_answers: list[str] | None = None,
    fuzzy_threshold: float = 0.6,
    suspicion_threshold: float = 0.4,
    log_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    aime_norm = [normalize_problem(p) for p in aime_problems]
    aime_norm_set = set(aime_norm)
    aime_shingles = [shingles(p) for p in aime_norm]

    kept = []
    drops = {"stage_2": 0, "stage_3": 0, "stage_4": 0}

    log_f = open(log_path, "w") if log_path else None
    try:
        for idx, row in enumerate(rows):
            p_norm = normalize_problem(row["problem"])

            if p_norm in aime_norm_set:
                aime_idx = aime_norm.index(p_norm)
                if log_f:
                    log_f.write(json.dumps({
                        "row_idx": idx, "source": row.get("source"),
                        "match_stage": 2, "aime_idx": aime_idx,
                        "problem": p_norm[:200],
                    }) + "\n")
                drops["stage_2"] += 1
                continue

            p_shingles = shingles(p_norm)
            best_j = 0.0
            best_aime = -1
            for i, sh in enumerate(aime_shingles):
                j = jaccard(p_shingles, sh)
                if j > best_j:
                    best_j = j
                    best_aime = i

            if best_j >= fuzzy_threshold:
                if log_f:
                    log_f.write(json.dumps({
                        "row_idx": idx, "source": row.get("source"),
                        "match_stage": 3, "aime_idx": best_aime,
                        "jaccard": round(best_j, 3),
                        "problem": p_norm[:200],
                    }) + "\n")
                drops["stage_3"] += 1
                continue

            if (
                best_j >= suspicion_threshold
                and aime_answers is not None
                and best_aime >= 0
                and str(row.get("answer", "")).strip() == str(aime_answers[best_aime]).strip()
            ):
                if log_f:
                    log_f.write(json.dumps({
                        "row_idx": idx, "source": row.get("source"),
                        "match_stage": 4, "aime_idx": best_aime,
                        "jaccard": round(best_j, 3),
                        "problem": p_norm[:200],
                    }) + "\n")
                drops["stage_4"] += 1
                continue

            kept.append(row)
    finally:
        if log_f:
            log_f.close()

    stats = dict(drops)
    stats["total_dropped"] = sum(drops.values())
    stats["total_kept"] = len(kept)
    stats["fuzzy_threshold"] = fuzzy_threshold
    stats["suspicion_threshold"] = suspicion_threshold
    return kept, stats
