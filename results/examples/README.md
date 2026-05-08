# Eval Output Examples

Each file is a full evaluation run output (50 AMC12 problems, or 723 for the MATH benchmark). Files cover the key experiments from `code/colab.ipynb` — one per major result in the paper.

## JSON Schema

```json
{
  "checkpoint": "peft_dora_train_heavy",  // run identifier
  "benchmark": "amc122024",              // amc122024 | amc122025 | aime2024 | math
  "scoring":   "greedy",                 // greedy | logit | twopass
  "n_samples": 1,                        // 1=greedy, >1=majority-vote
  "temperature": 0.0,
  "n_correct": 23,
  "n_total":   50,
  "accuracy":  0.46,
  "problems": [
    {
      "problem_id": "2024A-P05",
      "answer":     "C",                 // ground truth (A–E for AMC12)
      "predicted":  ["C"],               // model's prediction(s)
      "responses":  ["... reasoning chain ..."],
      "correct":    [true],
      "majority_correct": true
    }
  ]
}
```

## Scoring Modes

- **logit** — appends `"\nThe answer is ("` to the prompt, one forward pass, argmax over A–E token logits. Pure knowledge probe: does the model assign highest probability to the right answer?
- **greedy** — generates full response (up to 2048 tokens), parses `\boxed{}` or final letter. Measures generation quality.
- **twopass** — generates a reasoning chain first, then re-reads logits. Intermediate between logit and greedy.

## Key Example Pairs

Compare `dora_light_amc2024_logit.json` (38%) with `dora_light_amc2024_greedy.json` (12%) on the same problems to see the "knows but can't say" phenomenon: DoRA assigns high probability to the correct answer token but generates an incorrect response.

Compare `dora_heavy_amc2024_greedy.json` (46%) with `dora_light_amc2024_greedy.json` (12%) to see the effect of 3.7× more training data on DoRA's generation quality.
