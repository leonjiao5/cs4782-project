# DoRA for AIME

Re-implementation of DoRA (Liu et al., 2024) applied to AIME 2024 / 2025 competition math. CS 4782, Spring 2026.

## Introduction

Class project re-implementing DoRA — Weight-Decomposed Low-Rank Adaptation — and using it to fine-tune an LLM for AIME problems. Compared against a LoRA baseline.

## Chosen Result

DoRA's accuracy gain over LoRA at matched parameter budgets (Liu et al. 2024, Table 1). Our metric: AIME 2024 + 2025 accuracy.

## GitHub Contents

- `code/` — DoRA layer, model wiring, train/eval scripts
- `data/` — AIME 2024 / 2025 + training corpus (download instructions inside)
- `results/` — figures, tables, logs
- `poster/`, `report/` — final deliverables

## Re-implementation Details

Custom `DoRALinear` in PyTorch (we don't use `peft`'s `use_dora` flag). LoRA baseline uses `peft`. Base model TBD — see `TODO.txt`.

## Reproduction Steps

```
pip install -r requirements.txt
python code/train.py --method dora
python code/eval.py --checkpoint <path> --benchmark aime2024
```

GPU required.

## Results / Insights

TBD.

## Conclusion

TBD.

## References

- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, 2024.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.

## Acknowledgements

CS 4782, Cornell, Spring 2026.
