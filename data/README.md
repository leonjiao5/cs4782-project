# Data

## AIME 2024 / 2025

Pulled from HuggingFace via the `datasets` library. Both years are 30 problems (AIME I + II combined).

- AIME 2024: [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) — fields: `ID`, `Problem`, `Solution`, `Answer`
- AIME 2025: [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025) — fields: `problem_idx`, `problem`, `answer`, `problem_type`

We only use the problem and answer for evaluation. The loader in `code/data.py` normalizes both to lowercase `{problem, answer}` and writes one JSON per problem into `aime_2024/` or `aime_2025/`:

```
{"problem": "...", "answer": "42"}
```

## Training corpus

TBD — likely NuminaMath-CoT or OpenR1-Math-220k. Lands in `train/`.
