#!/usr/bin/env python3
"""Generate data, train a diffusion model, and save inference samples."""

import argparse
import sys
from pathlib import Path

from colab_train import build_env, find_latest_dataset, run_command


def find_latest_run(run_output: Path) -> Path:
    candidates = sorted(
        [path for path in run_output.glob("diffusion_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No diffusion run directory found under {run_output}")
    return candidates[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab-friendly DeepSculpt diffusion training pipeline"
    )
    parser.add_argument("--data-output", required=True, help="Base directory for generated training data")
    parser.add_argument("--run-output", required=True, help="Base directory for models and inference outputs")
    parser.add_argument("--data-mode", default="reuse", choices=["reuse", "regenerate"], help="Reuse the latest existing dataset or generate a new dated dataset version")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of generated training samples")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--void-dim", type=int, default=32, help="Voxel dimension")
    parser.add_argument("--timesteps", type=int, default=100, help="Diffusion timesteps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Diffusion learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Diffusion weight decay")
    parser.add_argument("--noise-schedule", default="cosine", choices=["linear", "cosine", "sigmoid"], help="Noise schedule")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Noise schedule beta start")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Noise schedule beta end")
    parser.add_argument("--num-inference-samples", type=int, default=1, help="Number of samples to infer after training")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps for sample generation")
    parser.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim", "dpm_solver"], help="Diffusion inference sampler")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale for sampling")
    parser.add_argument("--num-workers", type=int, default=0, help="Training dataloader workers")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay")
    parser.add_argument("--color", action="store_true", help="Enable 6-channel OHE color mode")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision for diffusion training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose CLI output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_output = Path(args.data_output).resolve()
    run_output = Path(args.run_output).resolve()
    data_output.mkdir(parents=True, exist_ok=True)
    run_output.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = run_output / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(repo_root, mpl_config_dir)

    global_flags = []
    if args.cpu:
        global_flags.append("--cpu")
    if args.verbose:
        global_flags.append("--verbose")

    base_cmd = [sys.executable, "-m", "deepsculpt.main", *global_flags]

    dataset_path = find_latest_dataset(data_output) if args.data_mode == "reuse" else None
    if dataset_path is None:
        generate_cmd = [
            *base_cmd,
            "generate-data",
            f"--num-samples={args.num_samples}",
            f"--void-dim={args.void_dim}",
            f"--output-dir={data_output}",
        ]
        run_command(generate_cmd, repo_root, env)
        dataset_path = find_latest_dataset(data_output)
    else:
        print(f"Reusing existing dataset at {dataset_path}")

    if dataset_path is None:
        raise FileNotFoundError(f"No generated dataset found under {data_output}")

    train_cmd = [
        *base_cmd,
        "train-diffusion",
        f"--data-folder={dataset_path}",
        f"--output-dir={run_output}",
        f"--epochs={args.epochs}",
        f"--batch-size={args.batch_size}",
        f"--void-dim={args.void_dim}",
        f"--timesteps={args.timesteps}",
        f"--learning-rate={args.learning_rate}",
        f"--weight-decay={args.weight_decay}",
        f"--noise-schedule={args.noise_schedule}",
        f"--beta-start={args.beta_start}",
        f"--beta-end={args.beta_end}",
        f"--num-workers={args.num_workers}",
        f"--ema-decay={args.ema_decay}",
        "--use-ema",
    ]
    if args.color:
        train_cmd.append("--color")
    if args.mixed_precision and not args.cpu:
        train_cmd.append("--mixed-precision")
    run_command(train_cmd, repo_root, env)

    latest_run = find_latest_run(run_output)
    checkpoint = latest_run / "diffusion_final.pt"
    inference_output = latest_run / "inference_samples"

    sample_cmd = [
        *base_cmd,
        "sample-diffusion",
        f"--checkpoint={checkpoint}",
        f"--num-samples={args.num_inference_samples}",
        f"--num-steps={args.num_inference_steps}",
        f"--sampler={args.sampler}",
        f"--guidance-scale={args.guidance_scale}",
        f"--output-dir={inference_output}",
        "--visualize",
    ]
    run_command(sample_cmd, repo_root, env)

    print(f"Training data used from: {dataset_path}")
    print(f"Run artifacts saved under: {latest_run}")
    print(f"Inference samples saved under: {inference_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
