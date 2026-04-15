from pathlib import Path

from scripts.colab_train import build_env, build_parser, find_generator_checkpoint, find_latest_dataset


def test_build_env_includes_repo_and_package_paths(tmp_path):
    repo_root = tmp_path / "repo"
    package_root = repo_root / "deepsculpt"
    mpl_dir = tmp_path / "mpl"
    package_root.mkdir(parents=True)
    mpl_dir.mkdir(parents=True)

    env = build_env(repo_root, mpl_dir)

    pythonpath_parts = env["PYTHONPATH"].split(":")
    assert str(repo_root) in pythonpath_parts
    assert str(package_root) in pythonpath_parts
    assert env["MPLCONFIGDIR"] == str(mpl_dir)


def test_find_generator_checkpoint_prefers_ema(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generator_ckpt = run_dir / "generator_final.pt"
    ema_ckpt = run_dir / "ema_generator_final.pt"
    generator_ckpt.write_text("raw")
    ema_ckpt.write_text("ema")

    checkpoint = find_generator_checkpoint(run_dir)

    assert checkpoint == ema_ckpt


def test_find_latest_dataset_uses_generated_metadata(tmp_path):
    older = tmp_path / "2026-04-14"
    newer = tmp_path / "2026-04-15"
    older.mkdir()
    newer.mkdir()
    (older / "dataset_metadata.json").write_text("{}")
    (newer / "dataset_metadata.json").write_text("{}")

    dataset = find_latest_dataset(tmp_path)

    assert dataset == newer


def test_find_latest_dataset_accepts_collection_metadata(tmp_path):
    dataset_dir = tmp_path / "2026-04-15"
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "collection_metadata.json").write_text("{}")

    dataset = find_latest_dataset(tmp_path)

    assert dataset == dataset_dir


def test_build_parser_uses_sparse_3d_safe_defaults():
    parser = build_parser()

    args = parser.parse_args([
        "--data-output", "/tmp/data",
        "--run-output", "/tmp/runs",
    ])

    assert args.r1_gamma == 2.0
    assert args.augment == "none"
    assert args.augment_target == 0.7
    assert args.occupancy_loss_weight == 5.0
    assert args.occupancy_floor == 0.01
    assert args.occupancy_target_mode == "batch_real"
