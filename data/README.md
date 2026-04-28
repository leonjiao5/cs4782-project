# Data

## Evaluation: AIME 2024 / 2025

Pulled from HuggingFace via the `datasets` library. Both years are 30 problems (AIME I + II combined).

- AIME 2024: [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) — fields: `ID`, `Problem`, `Solution`, `Answer`
- AIME 2025: [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025) — fields: `problem_idx`, `problem`, `answer`, `problem_type`

We only use problem + answer for evaluation. The loader in `code/data.py` normalizes both to lowercase `{problem, answer}` and writes one JSON per problem into `aime_2024/` or `aime_2025/`:

```
{"problem": "...", "answer": "42"}
```

## Training corpus

Shortlist from the project proposal (final pick TBD — see `../TODO.txt`):

- [HARP](https://arxiv.org/abs/2412.08819) — human-annotated math reasoning traces
- [`KbsdJames/Omni-MATH`](https://huggingface.co/datasets/KbsdJames/Omni-MATH) — ~4K Olympiad-level problems
- [`DigitalLearningGmbH/MATH-lighteval`](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) — ~12.5K competition problems with step-by-step solutions
- [`weijiezz/math-datasets-100k`](https://huggingface.co/datasets/weijiezz/math-datasets-100k) — 100k aggregated; contains a clean AIME25 split, so check for overlap before using
- [`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) — larger instruction-style fallback

Cross-check the training corpus against AIME 2024 / 2025 before training to avoid eval leakage.
