"""
Average adapter weights across selected checkpoints of a run.

Supports both PEFT adapters (adapter_model.safetensors) and custom DoRA (dora_adapters.pt).
Creates a new directory {run_dir}/checkpoint-avg{tag} with averaged weights + copied configs.

Usage:
    python code/avg_checkpoints.py --run_dir results/checkpoints/lora_train_light
    python code/avg_checkpoints.py --run_dir results/checkpoints/lora_train_light --steps 400 700
    python code/avg_checkpoints.py --run_dir results/checkpoints/lora_train_light --steps 400 700 --tag early
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
import torch


def avg_peft(run_dir: str, checkpoints: list[str], out_dir: str):
    from safetensors.torch import load_file, save_file

    print(f"  Averaging {len(checkpoints)} PEFT adapters → {out_dir}")
    accumulated = None
    for ckpt in checkpoints:
        weights = load_file(os.path.join(ckpt, "adapter_model.safetensors"))
        if accumulated is None:
            accumulated = {k: v.float() for k, v in weights.items()}
        else:
            for k in accumulated:
                accumulated[k] += weights[k].float()

    avg = {k: (v / len(checkpoints)).to(next(iter(
        load_file(os.path.join(checkpoints[0], "adapter_model.safetensors")).values()
    )).dtype) for k, v in accumulated.items()}

    os.makedirs(out_dir, exist_ok=True)
    save_file(avg, os.path.join(out_dir, "adapter_model.safetensors"))

    # Copy config files from first checkpoint
    for fname in ["adapter_config.json", "tokenizer_config.json", "tokenizer.json",
                  "chat_template.jinja", "training_args.bin", "README.md"]:
        src = os.path.join(checkpoints[0], fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))


def avg_dora(run_dir: str, checkpoints: list[str], out_dir: str):
    print(f"  Averaging {len(checkpoints)} custom DoRA adapters → {out_dir}")
    accumulated = None
    for ckpt in checkpoints:
        state = torch.load(os.path.join(ckpt, "dora_adapters.pt"),
                           map_location="cpu", weights_only=True)
        if accumulated is None:
            accumulated = {k: v.float() for k, v in state.items()}
        else:
            for k in accumulated:
                accumulated[k] += state[k].float()

    # Restore original dtype from first checkpoint
    ref = torch.load(os.path.join(checkpoints[0], "dora_adapters.pt"),
                     map_location="cpu", weights_only=True)
    avg = {k: (v / len(checkpoints)).to(ref[k].dtype) for k, v in accumulated.items()}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(avg, os.path.join(out_dir, "dora_adapters.pt"))

    # Copy dora_config.json from run root
    cfg = os.path.join(run_dir, "dora_config.json")
    if os.path.exists(cfg):
        shutil.copy2(cfg, os.path.join(out_dir, "dora_config.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="Path to checkpoint run dir (e.g. results/checkpoints/lora_train_light)")
    parser.add_argument("--steps", nargs="*", type=int, default=None,
                        help="Specific step numbers to average. Default: all checkpoints.")
    parser.add_argument("--tag", default="all",
                        help="Tag for the output dir name (default: 'all')")
    args = parser.parse_args()

    run_dir = args.run_dir
    all_ckpts = sorted(
        [d for d in os.listdir(run_dir)
         if d.startswith("checkpoint-") and d.split("-")[1].isdigit()],
        key=lambda x: int(x.split("-")[1])
    )
    if not all_ckpts:
        print(f"No checkpoints found in {run_dir}"); sys.exit(1)

    if args.steps:
        selected = [f"checkpoint-{s}" for s in args.steps]
        missing = [c for c in selected if c not in all_ckpts]
        if missing:
            print(f"WARNING: checkpoints not found: {missing}")
        selected = [c for c in selected if c in all_ckpts]
    else:
        selected = all_ckpts

    ckpt_paths = [os.path.join(run_dir, c) for c in selected]
    print(f"Run: {run_dir}")
    print(f"Averaging: {selected}")

    # Detect type from first checkpoint
    first = ckpt_paths[0]
    is_peft = os.path.exists(os.path.join(first, "adapter_model.safetensors"))
    # Custom DoRA saves adapters only at run root, not in intermediate checkpoints
    is_dora = os.path.exists(os.path.join(first, "dora_adapters.pt"))
    if not is_peft and not is_dora:
        print(f"ERROR: no adapter weights found in {first}")
        print("  (Custom DoRA saves adapters only at run root — skipping intermediate averaging)")
        sys.exit(1)

    out_name = f"checkpoint-avg_{args.tag}"
    out_dir = os.path.join(run_dir, out_name)

    if is_peft:
        avg_peft(run_dir, ckpt_paths, out_dir)
    elif is_dora:
        avg_dora(run_dir, ckpt_paths, out_dir)
    else:
        print(f"ERROR: could not detect checkpoint type in {first}"); sys.exit(1)

    print(f"  Saved → {out_dir}")


if __name__ == "__main__":
    main()
