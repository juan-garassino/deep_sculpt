# Training

This project supports two training modes: GAN and diffusion. Training runs in the GPU job container and logs to MLflow.

## Common Inputs

1. `TRAIN_REQUEST_JSON` can pass parameters as a single JSON string.
2. Environment variables override or provide defaults.

## GAN Training

Recommended default:
1. `training_mode=gan`
2. `model_variant=skip_color`

Example:
```bash
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{"params": {"training_mode": "gan", "model_variant": "skip_color", "epochs": 10}}'
```

Available variants:
1. `skip_mono`
2. `skip_color`
3. `complex_mono`
4. `complex_color`

## Diffusion Training

Recommended default:
1. `training_mode=diffusion`
2. `diffusion_model_type=unet3d`
3. `diffusion_schedule=cosine`
4. `diffusion_prediction_type=epsilon`
5. `color_mode=1`

Example:
```bash
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{"params": {"training_mode": "diffusion", "color_mode": 1, "diffusion_model_type": "unet3d", "diffusion_schedule": "cosine"}}'
```

Conditional diffusion:
1. `diffusion_model_type=conditional_unet3d`
2. `diffusion_conditioning_dim=512`
3. `diffusion_num_classes=<int>`
4. `diffusion_conditioning_mode=vector|one_hot|class_index`
5. `diffusion_conditioning_dropout=0.1` (for CFG)

## Outputs

1. MLflow run created with params and metrics.
2. TorchScript model uploaded to GCS.
3. `latest.json` pointer updated in GCS.
