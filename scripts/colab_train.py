#!/usr/bin/env python3
"""Generate data, train a monochrome GAN, and save inference samples."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd: Path) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_latest_run(run_output: Path) -> Path:
    candidates = sorted(
        [path for path in run_output.glob("gan_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No GAN run directory found under {run_output}")
    return candidates[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab-friendly DeepSculpt monochrome training pipeline"
    )
    parser.add_argument("--data-output", required=True, help="Base directory for generated training data")
    parser.add_argument("--run-output", required=True, help="Base directory for models and inference outputs")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of generated training samples")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--void-dim", type=int, default=32, help="Voxel dimension")
    parser.add_argument("--noise-dim", type=int, default=100, help="GAN noise dimension")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="GAN learning rate")
    parser.add_argument("--num-inference-samples", type=int, default=4, help="Number of samples to infer after training")
    parser.add_argument("--num-workers", type=int, default=0, help="Training dataloader workers")
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
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    global_flags = []
    if args.cpu:
        global_flags.append("--cpu")
    if args.verbose:
        global_flags.append("--verbose")

    base_cmd = [sys.executable, "-m", "deepsculpt.main", *global_flags]

    generate_cmd = [
        *base_cmd,
        "generate-data",
        f"--num-samples={args.num_samples}",
        f"--void-dim={args.void_dim}",
        f"--output-dir={data_output}",
    ]
    run_command(generate_cmd, repo_root)

    train_cmd = [
        *base_cmd,
        "train-gan",
        "--model-type=skip",
        f"--epochs={args.epochs}",
        f"--batch-size={args.batch_size}",
        f"--void-dim={args.void_dim}",
        f"--noise-dim={args.noise_dim}",
        f"--learning-rate={args.learning_rate}",
        f"--data-folder={data_output}",
        f"--output-dir={run_output}",
        f"--num-workers={args.num_workers}",
        "--generate-samples",
    ]
    if not args.cpu:
        train_cmd.append("--mixed-precision")
    run_command(train_cmd, repo_root)

    latest_run = find_latest_run(run_output)
    checkpoint = latest_run / "generator_final.pt"
    inference_output = latest_run / "inference_samples"

    sample_cmd = [
        *base_cmd,
        "sample-gan",
        f"--checkpoint={checkpoint}",
        f"--num-samples={args.num_inference_samples}",
        f"--output-dir={inference_output}",
        "--visualize",
    ]
    run_command(sample_cmd, repo_root)

    print(f"Training data saved under: {data_output}")
    print(f"Run artifacts saved under: {latest_run}")
    print(f"Inference samples saved under: {inference_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
