# Results

Aggregated results from our DoRA re-implementation on competition mathematics (AMC12 2024/2025 and MATH benchmark). See `report/report.pdf` for full analysis.

## Contents

### `results_table.csv`

Summary table across all main training runs. Columns:

| Column | Description |
|--------|-------------|
| `model` | Descriptive run name (e.g. "DoRA heavy (base)") |
| `method` | LoRA \| DoRA \| Baseline |
| `train_data` | Training tier: `train_light` (7.5K) \| `train_heavy` (27.5K) \| `train_s1k` (1K) |
| `rank` | LoRA/DoRA rank (r2, r4, r8, r32) |
| `base_model` | 3B-base or 3B-Instruct |
| `amc2024_greedy` | AMC12 2024 greedy accuracy (%) |
| `amc2024_logit` | AMC12 2024 logit accuracy (%) |
| `amc2024_twopass` | AMC12 2024 two-pass accuracy (%) |
| `amc2025_*` | Same for AMC12 2025 |
| `aime2024_*` | AIME 2024 greedy / maj-16 accuracy |
| `math_overall` | MATH benchmark overall accuracy (%) |
| `math_level1/3/5` | MATH accuracy by difficulty level |

### `figures/`

Four publication figures used in the 2-page report:

| File | Description |
|------|-------------|
| `figA_scoring_methods.png` | Overview of the three scoring modes (logit, two-pass, greedy) |
| `figG_math_by_level.png` | MATH benchmark accuracy by difficulty level (L1, L3, L5) |
| `figI_dora_vs_lora.png` | DoRA vs. LoRA comparison across scoring modes on AMC12 2024 |
| `figJ_logit_explainer.png` | Illustration of logit scoring — single forward pass vs. generation |

### `examples/`

Ten full eval output files, one per key experiment (50 AMC12 problems each; 723 for the MATH file). See `examples/README.md` for the JSON schema.

| File | Method | Benchmark | Scoring | Accuracy |
|------|--------|-----------|---------|---------|
| `baseline_amc2024_logit.json` | Baseline (3B-base) | AMC12 2024 | logit | 32% |
| `baseline_instruct_amc2024_greedy.json` | Baseline (3B-inst) | AMC12 2024 | greedy | 44% |
| `lora_light_amc2024_logit.json` | LoRA (base) | AMC12 2024 | logit | 38% |
| `lora_light_instruct_amc2024_logit.json` | LoRA (inst) | AMC12 2024 | logit | 44% |
| `lora_heavy_amc2024_greedy.json` | LoRA heavy (base) | AMC12 2024 | greedy | 36% |
| `dora_light_amc2024_logit.json` | DoRA (base) | AMC12 2024 | logit | 38% |
| `dora_light_amc2024_greedy.json` | DoRA (base) | AMC12 2024 | greedy | 12% |
| `dora_heavy_amc2024_greedy.json` | DoRA heavy (base) | AMC12 2024 | greedy | **46%** |
| `dora_r4_amc2024_logit.json` | DoRA r=4 (base) | AMC12 2024 | logit | **44%** |
| `math_lora_heavy_greedy.json` | LoRA heavy (base) | MATH | greedy | 35% |

The logit–greedy gap is the central finding: compare `dora_light_amc2024_logit.json` (38%) against `dora_light_amc2024_greedy.json` (12%) to see the model "knowing but not saying" the correct answer.
