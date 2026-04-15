#!/usr/bin/env python3
"""Summarize the latest DeepSculpt training run for autonomous experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _find_latest_run(run_output: Path, mode: str) -> Path:
    prefix = "gan_" if mode == "gan" else "diffusion_"
    candidates = sorted(
        [path for path in run_output.glob(f"{prefix}*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No {mode} run directory found under {run_output}")
    return candidates[-1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _gan_report(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "run_summary.json")
    config = _read_json(run_dir / "config.json")
    training_info = summary.get("training_info", {})
    last_epoch = summary.get("last_epoch_metrics", {}) or training_info.get("last_epoch_metrics", {})
    dataset_stats = summary.get("dataset_occupancy_stats", {})

    real_occ = float(last_epoch.get("real_occupancy", 0.0))
    fake_occ = float(last_epoch.get("fake_occupancy", 0.0))
    occ_gap = float(last_epoch.get("occupancy_gap", fake_occ - real_occ))
    collapse_events = float(training_info.get("collapse_events", 0.0))
    floor = float(config.get("training_params", {}).get("occupancy_floor", 0.01))
    collapsed = fake_occ < floor or collapse_events >= 3

    score = abs(occ_gap) + max(0.0, floor - fake_occ) * 10.0 + collapse_events
    suggested_status = "discard" if collapsed else "keep"
    summary_text = (
        f"GAN {'collapsed' if collapsed else 'healthy'}; "
        f"fake_occ={fake_occ:.4f}, real_occ={real_occ:.4f}, gap={occ_gap:.4f}, "
        f"collapse_events={collapse_events:.0f}"
    )

    return {
        "mode": "gan",
        "run_dir": str(run_dir),
        "score": round(score, 6),
        "suggested_status": suggested_status,
        "collapsed": collapsed,
        "summary": summary_text,
        "metrics": {
            "real_occupancy": real_occ,
            "fake_occupancy": fake_occ,
            "occupancy_gap": occ_gap,
            "occupancy_floor": floor,
            "collapse_events": collapse_events,
            "gen_loss": last_epoch.get("gen_loss"),
            "disc_loss": last_epoch.get("disc_loss"),
        },
        "dataset_path": summary.get("dataset_path"),
        "dataset_occupancy_stats": dataset_stats,
        "checkpoints": {
            "ema_generator": str(run_dir / "ema_generator_final.pt"),
            "generator": str(run_dir / "generator_final.pt"),
            "discriminator": str(run_dir / "discriminator_final.pt"),
        },
        "samples_dir": str(run_dir / "samples"),
        "inference_samples_dir": str(run_dir / "inference_samples"),
    }


def _diffusion_report(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "run_summary.json")
    config = _read_json(run_dir / "config.json")
    training_info = summary.get("training_info", {})
    last_epoch = summary.get("last_epoch_metrics", {}) or training_info.get("last_epoch_metrics", {})
    train_history = summary.get("train_history", {})

    diffusion_loss = last_epoch.get("diffusion_loss")
    if diffusion_loss is None:
        losses = train_history.get("train_loss", [])
        diffusion_loss = losses[-1] if losses else None
    score = float(diffusion_loss) if diffusion_loss is not None else 1e9
    suggested_status = "keep" if diffusion_loss is not None else "discard"
    summary_text = f"Diffusion final_loss={diffusion_loss:.6f}" if diffusion_loss is not None else "Diffusion summary missing loss"

    return {
        "mode": "diffusion",
        "run_dir": str(run_dir),
        "score": round(score, 6),
        "suggested_status": suggested_status,
        "summary": summary_text,
        "metrics": {
            "diffusion_loss": diffusion_loss,
            "mse_loss": last_epoch.get("mse_loss"),
            "l1_loss": last_epoch.get("l1_loss"),
        },
        "dataset_path": summary.get("dataset_path"),
        "dataset_occupancy_stats": summary.get("dataset_occupancy_stats", {}),
        "config": config,
        "checkpoint": str(run_dir / "diffusion_final.pt"),
        "inference_samples_dir": str(run_dir / "inference_samples"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the latest DeepSculpt run.")
    parser.add_argument("--run-output", required=True, help="Root directory containing gan_* or diffusion_* runs")
    parser.add_argument("--mode", required=True, choices=["gan", "diffusion"], help="Which training mode to summarize")
    parser.add_argument(
        "--fail-on-bad-status",
        action="store_true",
        help="Exit non-zero when the latest run should be discarded",
    )
    args = parser.parse_args()

    run_output = Path(args.run_output).resolve()
    run_dir = _find_latest_run(run_output, args.mode)
    report = _gan_report(run_dir) if args.mode == "gan" else _diffusion_report(run_dir)
    print(json.dumps(report, indent=2))
    if args.fail_on_bad_status and report.get("suggested_status") != "keep":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
