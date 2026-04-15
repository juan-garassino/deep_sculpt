# DeepSculpt Autoresearch

This folder adapts the `autoresearch` / RunPod pattern to DeepSculpt.

It is not a literal copy of the smaller reference project. DeepSculpt has:
- multiple training entrypoints
- procedural data generation
- heavier artifact output
- no single scalar GAN quality metric you can trust by default

So this version gives the agent:
- a container with Claude Code installed
- a DeepSculpt-specific `program.md`
- a stable evaluator script at `scripts/autoresearch_report.py`

## What the agent optimizes

The loop can optimize either:
- monochrome GAN training via `make colab-train-mono`
- diffusion training via `make colab-train-diffusion`

Environment variables choose the default mode and experiment budget.

## Build

```bash
make autoresearch-build
```

Or directly:

```bash
docker build -f autoresearch/Dockerfile -t deepsculpt-autoresearch .
```

## Run locally

```bash
docker run --gpus all \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e AUTORESEARCH_MODE=gan \
  -e AUTORESEARCH_DATA_OUT=/workspace/autoresearch_data \
  -e AUTORESEARCH_RUN_OUT=/workspace/autoresearch_runs \
  deepsculpt-autoresearch
```

## Run on RunPod

1. Build and push the image.
2. Create a GPU pod.
3. Set the image to your pushed `deepsculpt-autoresearch`.
4. Provide:
   - `ANTHROPIC_API_KEY`
   - optional `AUTORESEARCH_MODE=gan` or `diffusion`
   - optional data/run output overrides
5. Start the pod.

## Default behavior

- The entrypoint launches Claude Code automatically.
- Claude reads `autoresearch/program.md`.
- It uses `results.tsv` to log experiments.
- It scores the latest run with `scripts/autoresearch_report.py`.

## Important note

For GAN, this loop can compare runs and detect obvious occupancy collapse, but it still cannot replace human visual judgment entirely. The helper script gives a stable score and summary, not a perfect aesthetic metric.
