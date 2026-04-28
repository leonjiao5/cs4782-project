# DoRA for AIME

Re-implementation of DoRA (Liu et al., ICML 2024) applied to AIME 2024 / 2025 competition math. CS 4782, Cornell, Spring 2026.

**Authors:** Ryan Ye (rmy43), Boaz Ng (bn229), Leon Jiao (lsj47), Aadi Singla (ans262)

## Introduction

We re-implement DoRA — Weight-Decomposed Low-Rank Adaptation — and use it to fine-tune Llama (7B / 8B) on competition math. Original paper: [DoRA, Liu et al., ICML 2024](https://arxiv.org/abs/2402.09353).

## Chosen Result

The paper's main claim that DoRA outperforms LoRA at a matched parameter budget, plus the parameter-efficiency claim that DoRA at half rank tracks full-rank LoRA (Liu et al. 2024, Table 1 / Fig. 2). We measure both on AIME 2024 and AIME 2025 accuracy.

## GitHub Contents

- `code/` — DoRA layer, model wiring, train/eval scripts
- `data/` — AIME 2024 / 2025 eval sets + training-corpus instructions
- `results/` — figures, tables, logs
- `poster/`, `report/` — final deliverables

## Re-implementation Details

Custom `DoRALinear` in PyTorch. Primary base model is Llama (7B / 8B) per the proposal, with Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-7B as alternates. Baselines are stock LoRA (`peft`) and `peft`'s built-in DoRA (sanity check). Training corpus drawn from HARP / Omni-Math / MATH; eval is AIME 2024 + AIME 2025.

## Reproduction Steps

```
pip install -r requirements.txt
python code/train.py --method dora
python code/eval.py --checkpoint <path> --benchmark aime2024
```

Runs on Colab. A100 / H100 in bf16; T4 (16GB) requires 4-bit quantization via `bitsandbytes`.

## Results / Insights

TBD.

## Conclusion

TBD.

## References

- Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.

## Acknowledgements

Final project for CS 4782 (Deep Learning), Cornell University, Spring 2026.
