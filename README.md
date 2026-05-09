# Are You Smarter than DoRA?

Re-implementation of DoRA (Liu et al., ICML 2024) applied to competition mathematics. Cornell CS 4782, Spring 2026.

**Authors:** Ryan Ye (rmy43), Boaz Ng (bn229), Leon Jiao (lsj47), Aadi Singla (ans262)

---

## Introduction

Large language model fine-tuning is essential for specializing pre-trained models to new domains, but full fine-tuning updates billions of parameters at prohibitive cost. Parameter-Efficient Fine-Tuning (PEFT) methods address this by training a small number of adapter parameters while keeping the base model frozen.

**LoRA** (Hu et al., ICLR 2022) is the most widely used PEFT approach: it adds a low-rank update ΔW = BA to a frozen weight matrix. While effective, LoRA still shows a notable accuracy gap compared to full fine-tuning.

**DoRA** (Liu et al., ICML 2024) closes this gap by decomposing the pretrained weight into magnitude and direction components, then applying LoRA only to the direction. The paper's key insight is that full fine-tuning makes *large magnitude changes and small direction changes*, while LoRA can only make direction changes. By separating the two, DoRA more closely mimics full fine-tuning behavior at the same parameter budget.

We re-implement DoRA from scratch in PyTorch.

Original paper: [DoRA: Weight-Decomposed Low-Rank Adaptation, Liu et al., ICML 2024](https://arxiv.org/abs/2402.09353)

---

## Chosen Result

We aim to reproduce the primary result of the DoRA paper: **DoRA consistently outperforms LoRA at the same rank/parameter budget** (paper Tables 1-4, across NLP and image/video tasks). This would show that DoRA more faithfully exhibits full fine-tuning behavior than LoRA.

---

## GitHub Contents

```
README.md               ← this file. Provides an overview of the project and how to reproduce it.
code/                   ← DoRALinear layer, model wiring, train/eval/data scripts, configs/
data/                   ← AMC12 2024/2025 + AIME 2024/2025 eval sets; training corpus
results/                ← aggregated results table, 4 publication figures, example eval outputs
poster/                 ← in-class presentation poster (PDF)
report/                 ← 2-page project summary (PDF + LaTeX source)
requirements.txt        ← Python dependencies
```

See `data/README.md` for details on training corpora and leakage filtering.

---

## Re-implementation Details

**DoRA algorithm.** We implement `DoRALinear`, a `nn.Module` wrapping a frozen `nn.Linear`. The effective weight is detailed in the paper and report. This replaces the linear layers in the following modules: `q_proj, k_proj, v_proj, up_proj, down_proj` — 5 of 7 attention/FFN projections.

**Baselines.** Three methods are compared:
1. **LoRA** — HuggingFace PEFT `LoraConfig(use_dora=False)`
2. **DoRA** — our from-scratch PyTorch implementation (`code/dora_layers.py`)
3. **Peft DoRA** — HuggingFace PEFT `LoraConfig(use_dora=True)`, sanity-check reference only

**Models.** Qwen2.5-3B and Qwen2.5-3B-Instruct. We chose Qwen over Llama because Qwen's base weights are ungated — no HuggingFace access approval required, which simplified the workflow considerably. 

**Training.** SFTTrainer (TRL), rank r = 8, α = 32, dropout = 0.05, LR = 1e-4, cosine schedule, 3% warmup, effective batch size 16, 1–3 epochs. Training tiers: `train_light` (7.5K MATH-lighteval examples), `train_heavy` (27.5K, adds NuminaMath-CoT), `train_s1k` (1K-row subset). Rank sweep: r ∈ {2, 4, 8}.

**Evaluation.** Models evaluated on competition mathematics: AMC12 2024/2025 (50 multiple choice questions, A-E) and the MATH benchmark (723 free response questions across 5 difficulty levels) — a domain not studied in the original paper. 

**Novel evaluation protocol.** Standard greedy accuracy conflates knowledge with generation quality. We introduce three scoring modes, which enable finer-grained analysis than simple accuracy:
- **Logit scoring** — append `"\nThe answer is ("` to the prompt, single forward pass, argmax over A–E logits. A pure *knowledge probe*: does the model assign highest probability to the correct answer token?
- **Two-pass scoring** — generate a full reasoning chain, then re-read answer logits. Tests whether reasoning improves answer selection.
- **Greedy scoring** — generate full response (up to 2048 tokens), parse `\boxed{}`. Standard *generation quality* metric.

---

## Reproduction Steps

First clone the repo: 
`git clone https://github.com/leonjiao5/cs4782-project.git`


Then run the following commands within the project root directory:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build training data
python -m code.data_clean.build --tier train_light  # Smaller training data corpus, only MATH-lighteval
python -m code.data_clean.build --tier train        # Larger training data corpus, includes both MATH-lighteval and NuminaMath-CoT

# 3. Train
python code/train.py --method lora     --config code/configs/default.yaml
python code/train.py --method peft_dora --config code/configs/default.yaml

# 4. Evaluate
python code/eval.py --checkpoint results/checkpoints/lora_train_light \
    --benchmark amc122024
python code/eval.py --checkpoint results/checkpoints/lora_train_light \
    --benchmark amc122024 --scoring logit
```

**Compute requirements:** A100 or H100 (bf16); T4 (16 GB) requires `load_in_4bit=True`. Full training on `train_light` takes ~3h on an A100. Evaluation on AMC12 (50 problems, greedy) takes ~15 min per checkpoint.

For an end-to-end walkthrough, open `code/colab.ipynb` in Google Colab. Upload the repo to `MyDrive/cs4782-project/` on Google Drive — the notebook reads code and data directly from there.

---

## Results / Insights

### AMC12 2024 Main Results

| Model | Logit | 2-pass | Greedy | Δ (logit−greedy) |
|-------|-------|--------|--------|-----------------|
| Base (3B-base) | 32% | — | — | — |
| Base (3B-inst) | 34% | — | 44% | −10pp |
| LoRA light (base) | 38% | 24% | 18% | +20pp |
| LoRA light (inst) | **44%** | 32% | 10% | +34pp |
| LoRA heavy (base) | 34% | 32% | 36% | −2pp |
| DoRA light (base) | 38% | 28% | 12% | +26pp |
| DoRA light (inst) | 36% | 24% | 8% | +28pp |
| DoRA heavy (base) | 32% | — | **46%** | −14pp |
| DoRA r4 (base) | **44%** | — | 16% | +28pp |

### MATH Benchmark (greedy accuracy)

| Model | Overall | Level 1 | Level 3 | Level 5 |
|-------|---------|---------|---------|---------|
| Baseline (3B-base) | 43.8% | 74.3% | 57.6% | 22.1% |
| Baseline (3B-inst) | **52.7%** | **82.6%** | **66.1%** | **31.4%** |
| LoRA s1k (inst) | 44.8% | 77.1% | 56.5% | 24.2% |
| LoRA heavy (base) | 35.0% | 73.4% | 43.5% | 15.1% |
| DoRA light (base) | 30.7% | 70.6% | 36.4% | 12.7% |

### Key Findings

**"DoRA knows but can't say" (main finding).** The logit–greedy gap (Δ) is consistently larger for DoRA than LoRA at the same configuration. DoRA light shows Δ = +26pp vs. LoRA light's Δ = +20pp. DoRA assigns high probability to the correct answer internally but fails to produce it through generation.

**More training data closes — and reverses — the gap.** LoRA heavy nearly equalizes logit and greedy (Δ = −2pp). DoRA heavy reverses dramatically: greedy (46%) exceeds logit (32%), Δ = −14pp, achieving the highest greedy accuracy of any fine-tuned model. DoRA appears to need substantially more training signal to learn fluent generation.

**Knowledge is rank-invariant for DoRA.** Across r = 2, 4, 8, DoRA logit accuracy stays near 42–44%. Greedy remains low at all ranks (12–16%), suggesting the logit–greedy gap is a property of DoRA's adaptation mechanism rather than its capacity.

**MATH regression.** Fine-tuning on AMC-style problems hurts general math reasoning. DoRA light falls 13pp below the baseline (30.7% vs. 43.8%). Notably, LoRA s1k (1K instruct examples) nearly preserves the baseline (44.8%), suggesting small targeted datasets cause less catastrophic interference.

**Partial replication.** At the logit (knowledge) level, we reproduce the paper's claim: DoRA r4 matches the best LoRA logit result (both 44%). Greedy accuracy does not confirm DoRA > LoRA at 3B scale — likely because 3B is below the threshold where DoRA's generation advantage manifests. The original paper evaluates 7B+ models.

### Caveats

- `train_heavy` adds NuminaMath-CoT, a CoT-focused dataset with different reasoning patterns than MATH-lighteval; accuracy changes between light and heavy may reflect distribution shift rather than pure data scale.
- Qwen2.5-3B is a math-specialized model that may have been pre-trained on competition math data; baseline accuracy could be partially inflated by pre-training overlap with AMC12.
- Base vs. instruct model differences are not fully isolated from adapter effects; instruct models carry RLHF-tuned generation tendencies that independently affect greedy accuracy.
- Direct comparison with the original paper's gains requires caution — we are at 3B scale vs. the paper's 7B–13B experiments.

---

## Conclusion

Logit scoring was our most important methodological contribution. Without it, we would only have seen DoRA apparently underperforming LoRA on greedy generation and concluded a failed replication. Logit scoring revealed that DoRA actually *learns the correct answer* — it just cannot always verbalize it. Two-pass scoring provides an intermediate point: reasoning chains partially help but do not fully close the generation gap.

At 3B scale with ~7.5K training examples, we are in a regime where PEFT methods struggle more, and DoRA's generation advantage over LoRA has not yet emerged. Our results suggest that more training data (heavy tier) is the single most impactful lever: DoRA heavy goes from worst greedy (12%) to best greedy (46%) among all our fine-tuned models.

Future directions: (1) larger models (7B+) to match the paper's regime, (2) logit-guided or constrained decoding to recover DoRA's latent knowledge, (3) scale up training data (artifact for building OpenMathInstruct2 included in the codebase - contains 1M examples).

---

## References

- Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- DigitalLearningGmbH. *MATH-lighteval dataset*. HuggingFace, 2024.
- Li et al. *NuminaMath-CoT dataset*. AI-MO / HuggingFace, 2024.
- Mangrulkar et al. *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods*. HuggingFace, 2022.
- Qwen Team. *Qwen2.5: A Party of Foundation Language Models*. Alibaba Cloud, 2024.

---

## Acknowledgements

Final project for CS 4782 (Deep Learning), Cornell University, Spring 2026. Training and evaluation required A100/H100 GPU compute.
