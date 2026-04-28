import argparse


def main(args):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", choices=["dora", "lora", "full", "none"], default="dora")
    parser.add_argument("--output_dir", default="results/checkpoints")
    main(parser.parse_args())
