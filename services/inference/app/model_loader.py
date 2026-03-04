import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
from google.cloud import storage

import torch


@dataclass
class ModelBundle:
    model: torch.jit.ScriptModule
    run_id: Optional[str]
    artifact_uri: Optional[str]
    loaded_at: str
    noise_dim: int = 64
    out_dim: int = 16
    training_mode: str = "gan"
    diffusion_schedule: str = "cosine"
    diffusion_timesteps: int = 1000
    diffusion_prediction_type: str = "epsilon"
    in_channels: int = 1
    diffusion_model_type: str = "unet3d"
    diffusion_conditioning_dim: int = 512
    diffusion_num_classes: Optional[int] = None


_MODEL_CACHE: Optional[ModelBundle] = None


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    without_scheme = uri[len("gs://") :]
    bucket, blob = without_scheme.split("/", 1)
    return bucket, blob


def _download_gcs_file(gcs_uri: str, local_path: str) -> None:
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gcs_uri}")
    blob.download_to_filename(local_path)


def _read_pointer(pointer_uri: str) -> Dict[str, Any]:
    bucket_name, blob_name = _parse_gcs_uri(pointer_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return {}
    content = blob.download_as_text()
    return json.loads(content)


def load_latest_model(pointer_uri: str) -> ModelBundle:
    global _MODEL_CACHE

    pointer = _read_pointer(pointer_uri)
    artifact_uri = pointer.get("artifact_uri")
    if not artifact_uri:
        raise RuntimeError("latest model pointer missing artifact_uri")

    run_id = pointer.get("run_id")
    noise_dim = int(pointer.get("noise_dim", 100))
    out_dim = int(pointer.get("void_dim", pointer.get("out_dim", 64)))
    training_mode = str(pointer.get("training_mode", "gan")).lower()
    diffusion_model_type = str(pointer.get("diffusion_model_type", "unet3d"))
    diffusion_schedule = str(pointer.get("diffusion_schedule", "cosine"))
    diffusion_timesteps = int(pointer.get("diffusion_timesteps", 1000))
    diffusion_prediction_type = str(pointer.get("diffusion_prediction_type", "epsilon"))
    diffusion_conditioning_dim = int(pointer.get("diffusion_conditioning_dim", 512))
    diffusion_num_classes = pointer.get("diffusion_num_classes")
    if diffusion_num_classes is not None:
        diffusion_num_classes = int(diffusion_num_classes)
    in_channels = 1 if int(pointer.get("color_mode", 0)) == 0 else 6

    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_path = "/tmp/model.pt"
    _download_gcs_file(artifact_uri, local_path)
    model = torch.jit.load(local_path, map_location=device)
    model.eval()

    bundle = ModelBundle(
        model=model,
        run_id=run_id,
        artifact_uri=artifact_uri,
        loaded_at=datetime.now(timezone.utc).isoformat(),
        noise_dim=noise_dim,
        out_dim=out_dim,
        training_mode=training_mode,
        diffusion_model_type=diffusion_model_type,
        diffusion_schedule=diffusion_schedule,
        diffusion_timesteps=diffusion_timesteps,
        diffusion_prediction_type=diffusion_prediction_type,
        diffusion_conditioning_dim=diffusion_conditioning_dim,
        diffusion_num_classes=diffusion_num_classes,
        in_channels=in_channels,
    )
    _MODEL_CACHE = bundle
    return bundle


def get_cached_model() -> Optional[ModelBundle]:
    return _MODEL_CACHE


def run_inference(
    bundle: ModelBundle,
    input_vector: Optional[np.ndarray],
    num_samples: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    conditioning: Optional[np.ndarray] = None,
    class_label: Optional[int] = None,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if bundle.training_mode == "diffusion":
        from services.inference.app.diffusion_sampler import NoiseScheduler, sample_diffusion

        scheduler = NoiseScheduler(
            schedule_type=bundle.diffusion_schedule,
            timesteps=bundle.diffusion_timesteps,
            device=device,
        )
        conditioning_tensor = None
        class_labels = None
        if conditioning is not None:
            conditioning_tensor = torch.tensor(conditioning, dtype=torch.float32, device=device).view(
                -1, bundle.diffusion_conditioning_dim
            )
        if class_label is not None:
            if bundle.diffusion_num_classes is None:
                raise ValueError("class_label provided but model has no num_classes")
            class_labels = torch.full((num_samples,), int(class_label), device=device, dtype=torch.long)

        with torch.no_grad():
            output = sample_diffusion(
                model=bundle.model,
                scheduler=scheduler,
                shape=(
                    num_samples,
                    bundle.in_channels,
                    bundle.out_dim,
                    bundle.out_dim,
                    bundle.out_dim,
                ),
                device=device,
                num_inference_steps=min(num_inference_steps, bundle.diffusion_timesteps),
                conditioning=conditioning_tensor,
                class_labels=class_labels,
                guidance_scale=float(guidance_scale),
            )
        return output.detach().cpu().numpy()

    if input_vector is None:
        noise = torch.randn(num_samples, bundle.noise_dim, device=device)
    else:
        if input_vector.shape[-1] != bundle.noise_dim:
            raise ValueError(f"Expected noise_dim={bundle.noise_dim}, got {input_vector.shape[-1]}")
        noise = torch.tensor(input_vector, dtype=torch.float32, device=device).view(-1, bundle.noise_dim)

    with torch.no_grad():
        output = bundle.model(noise)

    return output.detach().cpu().numpy()


def run_latent_walk(bundle: ModelBundle, steps: int) -> list[np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = torch.randn(1, bundle.noise_dim, device=device)
    end = torch.randn(1, bundle.noise_dim, device=device)

    volumes = []
    with torch.no_grad():
        for i in range(steps):
            alpha = i / max(steps - 1, 1)
            noise = (1 - alpha) * start + alpha * end
            output = bundle.model(noise)
            volumes.append(output.detach().cpu().numpy())
    return volumes


    
