# Data

## Evaluation: AMC12 2024 / 2025

Multiple-choice problems (A–E answers) from the official MAA AMC12 exams. Our primary evaluation benchmark — more problems than AIME and well-suited for measuring adapter quality via logit scoring.

- AMC12 2024: 50 problems (`data/amc12_2024/`)
- AMC12 2025: 49 problems (`data/amc12_2025/`)

Schema (one JSON per problem):
```
{"problem": "...", "answer": "A", "difficulty": <int>}
```

Each problem is stored as a separate JSON file named by exam + problem number (e.g., `2024A-P10.json`). Answers are single letters A–E.

## Evaluation: MATH Benchmark (Levels 1, 3, 5)

Free-response competition math problems from `hendrycks/competition_mathematics` on HuggingFace. We use a filtered subset of 723 problems spanning difficulty levels 1, 3, and 5 across 7 subject areas (algebra, geometry, number theory, counting, precalculus, probability, intermediate algebra).

- File: `data/math_l135/math_l135.jsonl` (723 rows)

Schema:
```
{"id": "...", "problem": "...", "solution": "...", "answer": "...", "level": <1-5>, "type": "<subject>"}
```

Answers are free-response (extracted from `\boxed{...}` in solution). Used with greedy scoring only.

## Evaluation: AIME 2024 / 2025

Pulled from HuggingFace via the `datasets` library. Both years are 30 problems (AIME I + II combined).

- AIME 2024: [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) — fields: `ID`, `Problem`, `Solution`, `Answer`
- AIME 2025: [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025) — fields: `problem_idx`, `problem`, `answer`, `problem_type`

We only use problem + answer for evaluation. The downloader normalizes both to lowercase `{problem, answer}` and writes one JSON per problem into `aime_2024/` or `aime_2025/`:

```
{"problem": "...", "answer": "42"}
```

## Training corpora

Built by `code/data_clean/build.py`. Three tiers, one download per source (HuggingFace's cache prevents re-downloads across builds).

| Tier | Sources | Approx rows | Built |
|---|---|---|---|
| `train_light/` | `DigitalLearningGmbH/MATH-lighteval` (train split) | ~7,500 | yes |
| `train/` | MATH-lighteval + `AI-MO/NuminaMath-CoT` sub-sample | ~27,500 | yes |
| `train_scale/` | + `nvidia/OpenMathInstruct-2` sub-sample | TBD | deferred |

Each row in every tier follows the same on-disk schema:

```
{"problem": "...", "solution": "... \\boxed{ANSWER}", "answer": "ANSWER", "source": "<dataset>"}
```

Every solution is guaranteed to contain `\boxed{...}`, and `answer` is exactly the boxed content. This lets training and eval extract answers with the same regex.

### Building

```
python -m code.data_clean.build --tier {train_light,train,train_scale} \
    [--numinamath-sub-sample 20000] [--seed 42]
```

Output goes to `data/<tier>/<tier>.jsonl` plus a `leakage_report.json`. The larger training JSONLs aren't committed (they're regenerable; see `.gitignore`).

### Leakage filter

We want pre-2024 AIME / AMC / olympiad problems in training — they're the right kind of signal — but the 30 AIME 2024 + 30 AIME 2025 problems must stay out (they're our eval). We don't drop the `aops_forum` or `amc_aime` sources wholesale; instead, `code/data_clean/leakage.py` runs a multi-stage check against just the 60 AIME 2024 / 2025 problem strings:

1. **Exact match**: normalize whitespace + LaTeX wrappers, drop rows whose normalized problem is one of the 60 AIME problems.
2. **Fuzzy match**: drop rows with 5-shingle Jaccard ≥ 0.6 against any AIME problem.
3. **Answer cross-check**: for rows in the 0.4–0.6 Jaccard "suspicious" band, drop if the answer also matches the corresponding AIME answer.
4. **Audit log**: every dropped row is appended to `results/logs/leakage_drops.jsonl` for spot-checking.

Counts per stage land in `data/<tier>/leakage_report.json`. `train_light` skips this entirely (MATH-lighteval pre-dates AIME 2024/2025).

For our current `train/` build (seed=42, 20K NuminaMath sub-sample), the filter dropped 0 rows — none of the AIME 2024/2025 problems happened to appear. We verified the filter still works by injecting a verbatim AIME 2024 problem into the input; it was correctly dropped at stage 1.

### Sources we considered but dropped

- **HARP** — no public HF mirror found.
- **`KbsdJames/Omni-MATH`** — benchmark-only (test split). Reusing it as training is bad form.
- **`weijiezz/math-datasets-100k`** — only `question` + `answer`, no reasoning trace.
