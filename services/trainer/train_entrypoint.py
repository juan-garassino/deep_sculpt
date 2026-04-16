import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepsculpt.core.data.generation.pytorch_sculptor import PyTorchSculptor
from deepsculpt.core.models.model_factory import PyTorchModelFactory
from deepsculpt.core.training.gan_trainer import GANTrainer
from deepsculpt.core.training.diffusion_trainer import DiffusionTrainer
from deepsculpt.core.models.diffusion.unet import UNet3D, ConditionalUNet3D
from deepsculpt.core.models.diffusion.noise_scheduler import NoiseScheduler
from deepsculpt.core.training.base_trainer import TrainingConfig

from services.trainer.mlflow_utils import start_run, log_metric, end_run


def _is_local_mode() -> bool:
    bucket = os.environ.get("MODELS_BUCKET", "local")
    return not bucket or bucket == "local"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _get_required(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def load_train_params() -> Dict[str, Any]:
    raw = _env("TRAIN_REQUEST_JSON")
    params: Dict[str, Any] = {}
    if raw:
        try:
            params = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid TRAIN_REQUEST_JSON: {exc}") from exc

    model_variant = _env("MODEL_VARIANT")
    if model_variant:
        params.setdefault("model_variant", model_variant)

    params.setdefault("training_mode", _env("TRAINING_MODE", "diffusion"))
    params.setdefault("lr", float(_env("TRAIN_LR", "0.0002")))
    params.setdefault("epochs", int(_env("TRAIN_EPOCHS", "3")))
    params.setdefault("seed", int(_env("TRAIN_SEED", "42")))
    params.setdefault("dataset_id", _env("DATASET_ID", "synthetic"))
    params.setdefault("noise_dim", int(_env("NOISE_DIM", "100")))
    params.setdefault("void_dim", int(_env("VOID_DIM", "64")))
    params.setdefault("batch_size", int(_env("TRAIN_BATCH_SIZE", "4")))
    params.setdefault("model_type", _env("MODEL_TYPE", "skip"))
    params.setdefault("color_mode", int(_env("COLOR_MODE", "0")))
    params.setdefault("diffusion_model_type", _env("DIFFUSION_MODEL_TYPE", "unet3d"))
    params.setdefault("diffusion_schedule", _env("DIFFUSION_SCHEDULE", "cosine"))
    params.setdefault("diffusion_timesteps", int(_env("DIFFUSION_TIMESTEPS", "1000")))
    params.setdefault("diffusion_prediction_type", _env("DIFFUSION_PREDICTION_TYPE", "epsilon"))
    params.setdefault("diffusion_conditioning_dim", int(_env("DIFFUSION_CONDITIONING_DIM", "512")))
    params.setdefault("diffusion_num_classes", _env("DIFFUSION_NUM_CLASSES"))
    params.setdefault("diffusion_conditioning_mode", _env("DIFFUSION_CONDITIONING_MODE"))
    params.setdefault("diffusion_conditioning_dropout", float(_env("DIFFUSION_CONDITIONING_DROPOUT", "0.1")))
    params.setdefault("run_name", _env("RUN_NAME"))
    return params


class SyntheticSculptureDataset(Dataset):
    def __init__(
        self,
        length: int,
        void_dim: int,
        device: str,
        conditioning_mode: str | None = None,
        conditioning_dim: int = 512,
        num_classes: int | None = None,
    ):
        self.length = length
        self.void_dim = void_dim
        self.device = device
        self.conditioning_mode = conditioning_mode
        self.conditioning_dim = conditioning_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sculptor = PyTorchSculptor(void_dim=self.void_dim, device=self.device, sparse_mode=False, verbose=False)
        structure, colors = sculptor.generate_sculpture()
        if structure.is_sparse:
            structure = structure.to_dense()
        if colors.is_sparse:
            colors = colors.to_dense()
        sample = {
            "structure": structure.float(),
            "colors": colors.float(),
        }
        if self.conditioning_mode:
            if self.conditioning_mode == "vector":
                conditioning = torch.randn(self.conditioning_dim, device=self.device)
                sample["conditioning"] = conditioning
            elif self.conditioning_mode in {"one_hot", "class_index"}:
                if not self.num_classes:
                    raise ValueError("num_classes required for class_index/one_hot conditioning")
                class_label = torch.randint(0, self.num_classes, (1,), device=self.device).squeeze(0)
                sample["class_labels"] = class_label
                if self.conditioning_mode == "one_hot":
                    one_hot = torch.zeros(self.conditioning_dim, device=self.device)
                    idx = int(class_label.item()) % self.conditioning_dim
                    one_hot[idx] = 1.0
                    sample["conditioning"] = one_hot
            else:
                raise ValueError(f"Unknown conditioning_mode: {self.conditioning_mode}")

        return sample


def train_model(params: Dict[str, Any]) -> tuple[str, str]:
    torch.manual_seed(int(params["seed"]))
    np.random.seed(int(params["seed"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    variant = str(params.get("model_variant") or "").lower().strip()
    if variant:
        variant_map = {
            "skip_mono": ("skip", 0),
            "mono_skip": ("skip", 0),
            "skip_color": ("skip", 1),
            "color_skip": ("skip", 1),
            "complex_mono": ("complex", 0),
            "mono_complex": ("complex", 0),
            "complex_color": ("complex", 1),
            "color_complex": ("complex", 1),
        }
        if variant not in variant_map:
            raise ValueError(
                f"Unknown model_variant '{variant}'. "
                f"Supported: {', '.join(sorted(variant_map.keys()))}"
            )
        params["model_type"], params["color_mode"] = variant_map[variant]

    training_mode = str(params.get("training_mode", "gan")).lower()

    dataset = SyntheticSculptureDataset(
        length=max(8, int(params["batch_size"]) * 4),
        void_dim=int(params["void_dim"]),
        device=device,
        conditioning_mode=params.get("diffusion_conditioning_mode"),
        conditioning_dim=int(params.get("diffusion_conditioning_dim", 512)),
        num_classes=int(params["diffusion_num_classes"]) if params.get("diffusion_num_classes") else None,
    )
    loader = DataLoader(dataset, batch_size=int(params["batch_size"]), shuffle=True)

    if training_mode == "diffusion":
        in_channels = 1 if int(params["color_mode"]) == 0 else 6
        out_channels = in_channels
        if params["diffusion_model_type"] == "conditional_unet3d":
            num_classes = params.get("diffusion_num_classes")
            num_classes_val = int(num_classes) if num_classes else None
            model = ConditionalUNet3D(
                void_dim=int(params["void_dim"]),
                in_channels=in_channels,
                out_channels=out_channels,
                timesteps=int(params["diffusion_timesteps"]),
                conditioning_dim=int(params["diffusion_conditioning_dim"]),
                num_classes=num_classes_val,
            )
        else:
            model = UNet3D(
                void_dim=int(params["void_dim"]),
                in_channels=in_channels,
                out_channels=out_channels,
                timesteps=int(params["diffusion_timesteps"]),
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
        config = TrainingConfig(
            batch_size=int(params["batch_size"]),
            learning_rate=float(params["lr"]),
            epochs=int(params["epochs"]),
            mixed_precision=False,
            use_mlflow=False,
            use_wandb=False,
            use_tensorboard=False,
        )

        scheduler = NoiseScheduler(
            schedule_type=str(params["diffusion_schedule"]),
            timesteps=int(params["diffusion_timesteps"]),
            device=device,
        )
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=config,
            noise_scheduler=scheduler,
            device=device,
            prediction_type=str(params["diffusion_prediction_type"]),
            conditioning_key="conditioning" if params.get("diffusion_conditioning_mode") else None,
            conditioning_dropout=float(params.get("diffusion_conditioning_dropout", 0.1)),
        )

        for epoch in range(int(params["epochs"])):
            trainer.current_epoch = epoch
            metrics = trainer.train_epoch(loader)
            log_metric("diffusion_loss", float(metrics.get("diffusion_loss", 0.0)), step=epoch)
            log_metric("mse_loss", float(metrics.get("mse_loss", 0.0)), step=epoch)

        model.eval()
        model_path = "/tmp/model.pt"
        with torch.no_grad():
            example_x = torch.randn(
                1, in_channels, int(params["void_dim"]), int(params["void_dim"]), int(params["void_dim"]), device=device
            )
            example_t = torch.zeros(1, dtype=torch.long, device=device)
            if params["diffusion_model_type"] == "conditional_unet3d":
                example_c = torch.zeros(1, int(params["diffusion_conditioning_dim"]), device=device)
                if params.get("diffusion_num_classes"):
                    example_labels = torch.zeros(1, dtype=torch.long, device=device)
                    traced = torch.jit.trace(model, (example_x, example_t, example_c, example_labels))
                else:
                    traced = torch.jit.trace(model, (example_x, example_t, example_c))
            else:
                traced = torch.jit.trace(model, (example_x, example_t))
            traced.save(model_path)

        sample_path = "/tmp/sample.npy"
        with torch.no_grad():
            sample = trainer.pipeline.sample(
                shape=(1, in_channels, int(params["void_dim"]), int(params["void_dim"]), int(params["void_dim"]))
            )
            np.save(sample_path, sample.detach().cpu().numpy())

        return model_path, sample_path

    # GAN training
    factory = PyTorchModelFactory(device=device)
    generator = factory.create_gan_generator(
        model_type=params["model_type"],
        void_dim=int(params["void_dim"]),
        noise_dim=int(params["noise_dim"]),
        color_mode=int(params["color_mode"]),
        sparse=False,
    )
    discriminator = factory.create_gan_discriminator(
        model_type=params["model_type"],
        void_dim=int(params["void_dim"]),
        color_mode=int(params["color_mode"]),
        sparse=False,
    )

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=float(params["lr"]), betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(params["lr"]), betas=(0.5, 0.999))

    config = TrainingConfig(
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["lr"]),
        epochs=int(params["epochs"]),
        mixed_precision=False,
        use_mlflow=False,
        use_wandb=False,
        use_tensorboard=False,
    )

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        config=config,
        device=device,
        noise_dim=int(params["noise_dim"]),
        loss_type="bce",
        use_gradient_penalty=False,
    )

    for epoch in range(int(params["epochs"])):
        trainer.current_epoch = epoch
        metrics = trainer.train_epoch(loader)
        log_metric("gen_loss", float(metrics.get("gen_loss", 0.0)), step=epoch)
        log_metric("disc_loss", float(metrics.get("disc_loss", 0.0)), step=epoch)

    generator.eval()
    model_path = "/tmp/model.pt"
    with torch.no_grad():
        example = torch.randn(1, int(params["noise_dim"]), device=device)
        traced = torch.jit.trace(generator, example)
        traced.save(model_path)

    sample_path = "/tmp/sample.npy"
    with torch.no_grad():
        sample_noise = torch.randn(1, int(params["noise_dim"]), device=device)
        sample_out = generator(sample_noise).detach().cpu().numpy()
        np.save(sample_path, sample_out)

    return model_path, sample_path


def upload_to_gcs(local_path: str, gcs_uri: str) -> str:
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")
    without_scheme = gcs_uri[len("gs://") :]
    bucket_name, blob_path = without_scheme.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return gcs_uri


def _save_local(model_path: str, sample_path: str, run_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Save model and pointer to the local /app/models volume."""
    models_dir = Path(os.environ.get("LOCAL_MODELS_DIR", "/app/models"))
    run_dir = models_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    local_model = str(run_dir / "model.pt")
    local_sample = str(run_dir / "sample.npy")
    shutil.copy2(model_path, local_model)
    shutil.copy2(sample_path, local_sample)

    pointer_payload = {
        "run_id": run_id,
        "artifact_uri": local_model,
        "mlflow_run_uri": f"runs:/{run_id}/model",
        "noise_dim": int(params["noise_dim"]),
        "void_dim": int(params["void_dim"]),
        "model_type": params["model_type"],
        "color_mode": int(params["color_mode"]),
        "model_variant": params.get("model_variant"),
        "training_mode": params.get("training_mode"),
        "diffusion_model_type": params.get("diffusion_model_type"),
        "diffusion_schedule": params.get("diffusion_schedule"),
        "diffusion_timesteps": int(params.get("diffusion_timesteps", 1000)),
        "diffusion_prediction_type": params.get("diffusion_prediction_type"),
        "diffusion_conditioning_dim": int(params.get("diffusion_conditioning_dim", 512)),
        "diffusion_num_classes": params.get("diffusion_num_classes"),
        "diffusion_conditioning_mode": params.get("diffusion_conditioning_mode"),
        "diffusion_conditioning_dropout": float(params.get("diffusion_conditioning_dropout", 0.1)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    pointer_path = models_dir / "latest.json"
    pointer_path.write_text(json.dumps(pointer_payload, indent=2))

    return pointer_payload


def _save_gcs(model_path: str, sample_path: str, run_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Upload model to GCS and write the remote pointer."""
    from services.trainer.gcs_model_pointer import write_pointer

    models_bucket = _get_required("MODELS_BUCKET")
    latest_pointer = _env("LATEST_MODEL_POINTER_PATH", f"gs://{models_bucket}/models/latest.json")

    model_gcs_uri = f"gs://{models_bucket}/models/{run_id}/model.pt"
    upload_to_gcs(model_path, model_gcs_uri)

    pointer_payload = {
        "run_id": run_id,
        "artifact_uri": model_gcs_uri,
        "mlflow_run_uri": f"runs:/{run_id}/model",
        "noise_dim": int(params["noise_dim"]),
        "void_dim": int(params["void_dim"]),
        "model_type": params["model_type"],
        "color_mode": int(params["color_mode"]),
        "model_variant": params.get("model_variant"),
        "training_mode": params.get("training_mode"),
        "diffusion_model_type": params.get("diffusion_model_type"),
        "diffusion_schedule": params.get("diffusion_schedule"),
        "diffusion_timesteps": int(params.get("diffusion_timesteps", 1000)),
        "diffusion_prediction_type": params.get("diffusion_prediction_type"),
        "diffusion_conditioning_dim": int(params.get("diffusion_conditioning_dim", 512)),
        "diffusion_num_classes": params.get("diffusion_num_classes"),
        "diffusion_conditioning_mode": params.get("diffusion_conditioning_mode"),
        "diffusion_conditioning_dropout": float(params.get("diffusion_conditioning_dropout", 0.1)),
    }
    write_pointer(latest_pointer, pointer_payload)
    pointer_payload["latest_pointer"] = latest_pointer
    return pointer_payload


def main() -> None:
    params = load_train_params()

    run = start_run(params)
    run_id = run.info.run_id

    try:
        model_path, sample_path = train_model(params)

        import mlflow
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(sample_path, artifact_path="samples")

        if _is_local_mode():
            pointer_payload = _save_local(model_path, sample_path, run_id, params)
        else:
            pointer_payload = _save_gcs(model_path, sample_path, run_id, params)

        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_id": run_id,
                    "mlflow_run_uri": pointer_payload["mlflow_run_uri"],
                    "artifact_uri": pointer_payload["artifact_uri"],
                },
                indent=2,
            )
        )
    finally:
        end_run()


if __name__ == "__main__":
    try:
        use_prefect = os.environ.get("USE_PREFECT_FLOW", "false").lower() == "true"
        if use_prefect:
            from deepsculpt.core.workflow.pytorch_workflow import build_pytorch_flow

            flow = build_pytorch_flow(framework="pytorch", training_mode="gan")
            flow.run(parameters={
                "epochs": int(os.environ.get("TRAIN_EPOCHS", "3")),
                "model_type": os.environ.get("MODEL_TYPE", "skip"),
            })
        else:
            main()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise
