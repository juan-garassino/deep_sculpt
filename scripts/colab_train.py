#!/usr/bin/env python3
"""Generate data, train a monochrome GAN, and save inference samples."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_env(repo_root: Path, mpl_config_dir: Path) -> dict:
    env = os.environ.copy()
    pythonpath_parts = [
        str(repo_root),
        str(repo_root / "deepsculpt"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    return env


def run_command(cmd, cwd: Path, env: dict) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def find_latest_run(run_output: Path) -> Path:
    candidates = sorted(
        [path for path in run_output.glob("gan_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No GAN run directory found under {run_output}")
    return candidates[-1]


def find_latest_dataset(data_output: Path) -> Path | None:
    candidates = []
    for path in data_output.iterdir():
        if not path.is_dir():
            continue
        if (path / "dataset_metadata.json").exists():
            candidates.append(path)
            continue
        if (path / "metadata" / "collection_metadata.json").exists():
            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_generator_checkpoint(run_dir: Path) -> Path:
    ema_checkpoint = run_dir / "ema_generator_final.pt"
    if ema_checkpoint.exists():
        return ema_checkpoint
    return run_dir / "generator_final.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab-friendly DeepSculpt monochrome training pipeline"
    )
    parser.add_argument("--data-output", required=True, help="Base directory for generated training data")
    parser.add_argument("--run-output", required=True, help="Base directory for models and inference outputs")
    parser.add_argument("--data-mode", default="reuse", choices=["reuse", "regenerate"], help="Reuse the latest existing dataset or generate a new dated dataset version")
    parser.add_argument("--structure-preset", default="architectural", choices=["architectural", "generic"], help="Procedural structure preset for generated data")
    parser.add_argument("--grid-count", type=int, default=1, help="Grid enable/count flag")
    parser.add_argument("--grid-step", type=int, default=4, help="Grid column spacing")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of generated training samples")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--void-dim", type=int, default=32, help="Voxel dimension")
    parser.add_argument("--noise-dim", type=int, default=100, help="GAN noise dimension")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="GAN learning rate")
    parser.add_argument("--num-inference-samples", type=int, default=4, help="Number of samples to infer after training")
    parser.add_argument("--num-workers", type=int, default=0, help="Training dataloader workers")
    parser.add_argument("--discriminator-type", default="spectral_norm", help="GAN discriminator type")
    parser.add_argument("--r1-gamma", type=float, default=2.0, help="R1 regularization gamma")
    parser.add_argument("--r1-interval", type=int, default=16, help="R1 regularization interval")
    parser.add_argument("--augment", default="none", choices=["none", "ada-lite"], help="Discriminator augmentation policy")
    parser.add_argument("--augment-p", type=float, default=0.0, help="Initial augmentation probability")
    parser.add_argument("--augment-target", type=float, default=0.7, help="Target real accuracy for ADA-lite")
    parser.add_argument("--occupancy-loss-weight", type=float, default=5.0, help="Weight for occupancy regularization")
    parser.add_argument("--occupancy-floor", type=float, default=0.01, help="Minimum healthy occupancy floor")
    parser.add_argument("--occupancy-target-mode", default="batch_real", choices=["batch_real", "dataset_mean"], help="Occupancy target source")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision for GAN training")
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
            f"--structure-preset={args.structure_preset}",
            f"--grid-count={args.grid_count}",
            f"--grid-step={args.grid_step}",
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
        "train-gan",
        "--model-type=skip",
        f"--epochs={args.epochs}",
        f"--batch-size={args.batch_size}",
        f"--void-dim={args.void_dim}",
        f"--noise-dim={args.noise_dim}",
        f"--learning-rate={args.learning_rate}",
        f"--data-folder={dataset_path}",
        f"--output-dir={run_output}",
        f"--num-workers={args.num_workers}",
        f"--discriminator-type={args.discriminator_type}",
        f"--r1-gamma={args.r1_gamma}",
        f"--r1-interval={args.r1_interval}",
        f"--augment={args.augment}",
        f"--augment-p={args.augment_p}",
        f"--augment-target={args.augment_target}",
        f"--occupancy-loss-weight={args.occupancy_loss_weight}",
        f"--occupancy-floor={args.occupancy_floor}",
        f"--occupancy-target-mode={args.occupancy_target_mode}",
        f"--ema-decay={args.ema_decay}",
        "--use-ema",
        "--sample-from-ema",
        "--generate-samples",
    ]
    if args.mixed_precision and not args.cpu:
        train_cmd.append("--mixed-precision")
    run_command(train_cmd, repo_root, env)

    latest_run = find_latest_run(run_output)
    checkpoint = find_generator_checkpoint(latest_run)
    inference_output = latest_run / "inference_samples"

    sample_cmd = [
        *base_cmd,
        "sample-gan",
        f"--checkpoint={checkpoint}",
        f"--num-samples={args.num_inference_samples}",
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
