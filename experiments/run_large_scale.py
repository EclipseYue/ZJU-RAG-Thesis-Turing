import argparse

from run_all import run_automated_ablation_with_tracking


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a large-scale real benchmark ablation.")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wiki"], help="Benchmark dataset name.")
    parser.add_argument("--split", default="validation", help="Dataset split.")
    parser.add_argument("--samples", type=int, default=500, help="Number of sampled benchmark queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval depth.")
    parser.add_argument("--device", default=None, help="Explicit runtime device, e.g. cuda or cpu.")
    parser.add_argument("--output-name", default="automated_ablation.json", help="Output matrix filename.")
    parser.add_argument("--mock", action="store_true", help="Force mock mode for quick dry runs.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases tracking.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    print(f"🚀 Running Large-Scale Ablation on {args.dataset} (N={args.samples})")
    run_automated_ablation_with_tracking(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.samples,
        top_k=args.top_k,
        use_wandb=args.use_wandb,
        force_mock=args.mock,
        device=args.device,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
