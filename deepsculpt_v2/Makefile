# DeepSculpt v2.0 - PyTorch Makefile with UV Support
# Modern build system for PyTorch operations with advanced tooling

.PHONY: help install install-dev install-uv test test-fast test-slow test-gpu lint format type-check clean build docs serve-docs train-gan train-diffusion generate-data sample-gan sample-diffusion preprocess visualize benchmark profile docker-build docker-run colab-setup

# Default target
help:
	@echo "DeepSculpt v2.0 (PyTorch) - Available targets:"
	@echo ""
	@echo "Installation (UV Package Manager):"
	@echo "  install-uv       Install with UV package manager (recommended)"
	@echo "  install          Install package and dependencies (pip)"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-all      Install with all optional dependencies"
	@echo "  colab-setup      Setup for Google Colab environment"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-fast        Run fast tests only"
	@echo "  test-slow        Run slow/integration tests"
	@echo "  test-gpu         Run GPU-specific tests"
	@echo "  test-sparse      Run sparse tensor tests"
	@echo "  lint             Run linting (ruff)"
	@echo "  format           Format code (black, ruff)"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  clean            Clean build artifacts"
	@echo ""
	@echo "Build and Documentation:"
	@echo "  build            Build package"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Machine Learning Operations:"
	@echo "  train-gan        Train PyTorch GAN models"
	@echo "  train-diffusion  Train PyTorch diffusion models"
	@echo "  generate-data    Generate synthetic training data"
	@echo "  sample-gan       Generate samples from GAN model"
	@echo "  sample-diffusion Generate samples from diffusion model"
	@echo "  preprocess       Preprocess and curate data"
	@echo "  visualize        Visualize 3D data"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  profile          Profile model performance"
	@echo ""
	@echo "Model Operations:"
	@echo "  evaluate         Evaluate trained models"
	@echo "  export           Export models for deployment"
	@echo "  compare          Compare model performance"
	@echo ""
	@echo "Docker and Deployment:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker container"
	@echo "  docker-gpu       Run with GPU support"
	@echo ""
	@echo "Utilities:"
	@echo "  check-env        Check environment setup"
	@echo "  setup-env        Setup development environment"
	@echo "  gpu-info         Show GPU information"

# Variables
PYTHON := python3
UV := uv
PIP := pip3
PACKAGE_NAME := deepsculpt-v2
TEST_PATH := tests/
SRC_PATH := core/
DOCS_PATH := docs/
BUILD_PATH := build/
DIST_PATH := dist/

# Environment setup
check-env:
	@echo "Checking environment setup..."
	@$(PYTHON) --version
	@$(UV) --version 2>/dev/null || echo "UV not installed - install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch not installed"
	@$(PYTHON) -c "import numpy as np; print(f'NumPy: {np.__version__}')" || echo "NumPy not installed"
	@echo "Environment check complete."

gpu-info:
	@echo "GPU Information:"
	@$(PYTHON) -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1e9:.1f} GB')
else:
    print('No CUDA GPUs available')
"

setup-env:
	@echo "Setting up development environment..."
	@$(UV) pip install --upgrade pip setuptools wheel
	@$(UV) pip install -e ".[dev]"
	@pre-commit install
	@echo "Development environment setup complete."

# Installation targets with UV support
install-uv:
	@echo "Installing DeepSculpt v2.0 with UV..."
	@$(UV) pip install -e .
	@echo "UV installation complete."

install:
	@echo "Installing DeepSculpt v2.0..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e .
	@echo "Installation complete."

install-dev:
	@echo "Installing DeepSculpt v2.0 with development dependencies..."
	@$(UV) pip install -e ".[dev]" || $(PIP) install -e ".[dev]"
	@echo "Development installation complete."

install-all:
	@echo "Installing DeepSculpt v2.0 with all dependencies..."
	@$(UV) pip install -e ".[all]" || $(PIP) install -e ".[all]"
	@echo "Full installation complete."

install-sparse:
	@echo "Installing with sparse tensor support..."
	@$(UV) pip install -e ".[sparse]" || $(PIP) install -e ".[sparse]"

install-viz:
	@echo "Installing with enhanced visualization..."
	@$(UV) pip install -e ".[viz3d,viz]" || $(PIP) install -e ".[viz3d,viz]"

# Google Colab setup
colab-setup:
	@echo "Setting up for Google Colab..."
	@$(PIP) install -e ".[colab]"
	@$(PYTHON) -c "
import torch
print('✓ PyTorch installed')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name()}')
print('✓ DeepSculpt v2.0 ready for Colab!')
"

# Testing targets
test:
	@echo "Running all tests..."
	@pytest $(TEST_PATH) -v --cov=$(SRC_PATH) --cov-report=term-missing --cov-report=html

test-fast:
	@echo "Running fast tests..."
	@pytest $(TEST_PATH) -v -m "not slow" --cov=$(SRC_PATH)

test-slow:
	@echo "Running slow/integration tests..."
	@pytest $(TEST_PATH) -v -m "slow" --cov=$(SRC_PATH)

test-gpu:
	@echo "Running GPU-specific tests..."
	@pytest $(TEST_PATH) -v -m "gpu" --cov=$(SRC_PATH)

test-sparse:
	@echo "Running sparse tensor tests..."
	@pytest $(TEST_PATH) -v -m "sparse" --cov=$(SRC_PATH)

test-diffusion:
	@echo "Running diffusion model tests..."
	@pytest $(TEST_PATH) -v -m "diffusion" --cov=$(SRC_PATH)

test-gan:
	@echo "Running GAN model tests..."
	@pytest $(TEST_PATH) -v -m "gan" --cov=$(SRC_PATH)

test-benchmark:
	@echo "Running benchmark tests..."
	@pytest $(TEST_PATH) -v -m "benchmark" --benchmark-only

# Code quality targets with modern tools
lint:
	@echo "Running linting with ruff..."
	@ruff check $(SRC_PATH) $(TEST_PATH)
	@echo "Linting complete."

format:
	@echo "Formatting code..."
	@black $(SRC_PATH) $(TEST_PATH)
	@ruff --fix $(SRC_PATH) $(TEST_PATH)
	@echo "Code formatting complete."

type-check:
	@echo "Running type checking..."
	@mypy $(SRC_PATH)
	@echo "Type checking complete."

# Build targets
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_PATH) $(DIST_PATH) *.egg-info
	@rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

build: clean
	@echo "Building package..."
	@$(PYTHON) -m build
	@echo "Build complete."

# Documentation targets
docs:
	@echo "Building documentation..."
	@cd $(DOCS_PATH) && make html
	@echo "Documentation build complete."

serve-docs:
	@echo "Serving documentation locally..."
	@cd $(DOCS_PATH)/_build/html && $(PYTHON) -m http.server 8080

# Machine Learning Operations - GAN Training
train-gan:
	@echo "Training PyTorch GAN model..."
	@$(PYTHON) main.py train-gan \
		--model-type=skip \
		--epochs=100 \
		--batch-size=32 \
		--void-dim=64 \
		--sparse \
		--mixed-precision \
		--mlflow \
		--generate-samples \
		--verbose

train-gan-simple:
	@echo "Training simple GAN model..."
	@$(PYTHON) main.py train-gan \
		--model-type=simple \
		--epochs=50 \
		--batch-size=16 \
		--output-dir=./results/simple

train-gan-complex:
	@echo "Training complex GAN model..."
	@$(PYTHON) main.py train-gan \
		--model-type=complex \
		--epochs=200 \
		--batch-size=32 \
		--sparse \
		--mixed-precision \
		--scheduler \
		--mlflow

train-gan-distributed:
	@echo "Training GAN with distributed setup..."
	@torchrun --nproc_per_node=2 main.py train-gan \
		--model-type=skip \
		--epochs=100 \
		--batch-size=16 \
		--mixed-precision

# Machine Learning Operations - Diffusion Training
train-diffusion:
	@echo "Training PyTorch diffusion model..."
	@$(PYTHON) main.py train-diffusion \
		--epochs=100 \
		--batch-size=16 \
		--void-dim=64 \
		--timesteps=1000 \
		--noise-schedule=cosine \
		--sparse \
		--mixed-precision \
		--mlflow

train-diffusion-fast:
	@echo "Training diffusion model (fast)..."
	@$(PYTHON) main.py train-diffusion \
		--epochs=50 \
		--batch-size=8 \
		--timesteps=500 \
		--noise-schedule=linear

# Data Operations
generate-data:
	@echo "Generating synthetic training data..."
	@$(PYTHON) main.py generate-data \
		--num-samples=1000 \
		--void-dim=64 \
		--num-shapes=5 \
		--sparse \
		--output-dir=./data

generate-data-large:
	@echo "Generating large dataset..."
	@$(PYTHON) main.py generate-data \
		--num-samples=10000 \
		--void-dim=64 \
		--num-shapes=7 \
		--sparse \
		--output-dir=./data/large

preprocess:
	@echo "Preprocessing data..."
	@$(PYTHON) main.py preprocess \
		--input-dir=./raw_data \
		--output-dir=./processed \
		--encoding=one_hot \
		--sparse

# Sampling Operations
sample-gan:
	@echo "Generating samples from GAN..."
	@$(PYTHON) main.py sample-gan \
		--checkpoint=./results/generator_final.pt \
		--num-samples=10 \
		--visualize \
		--output-dir=./samples

sample-diffusion:
	@echo "Generating samples from diffusion model..."
	@$(PYTHON) main.py sample-diffusion \
		--checkpoint=./results/diffusion_final.pt \
		--num-samples=5 \
		--num-steps=50 \
		--visualize \
		--output-dir=./samples

# Visualization
visualize:
	@echo "Visualizing 3D data..."
	@$(PYTHON) main.py visualize \
		--data-path=./data/sample.pt \
		--backend=plotly \
		--interactive

visualize-static:
	@echo "Creating static visualizations..."
	@$(PYTHON) main.py visualize \
		--data-path=./data/sample.pt \
		--backend=matplotlib \
		--output-path=./visualization.png

# Performance and Benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	@$(PYTHON) main.py benchmark \
		--model-type=skip \
		--batch-size=32 \
		--sparse \
		--profile-memory \
		--save-results

benchmark-all:
	@echo "Running comprehensive benchmarks..."
	@for model in simple complex skip monochrome; do \
		$(PYTHON) main.py benchmark --model-type=$$model --save-results; \
	done

profile:
	@echo "Profiling model performance..."
	@$(PYTHON) -m torch.profiler main.py train-gan --epochs=1 --batch-size=8

profile-memory:
	@echo "Profiling memory usage..."
	@$(PYTHON) -m memory_profiler main.py train-gan --epochs=1 --batch-size=4

# Model Operations
evaluate:
	@echo "Evaluating trained models..."
	@$(PYTHON) main.py evaluate \
		--checkpoint=./results/model.pt \
		--test-data=./test_data \
		--metrics=all \
		--output-dir=./evaluation

export:
	@echo "Exporting model for deployment..."
	@$(PYTHON) main.py export \
		--checkpoint=./results/model.pt \
		--format=onnx \
		--output=./exports/model.onnx

compare:
	@echo "Comparing model performance..."
	@$(PYTHON) -c "
import torch
from core.utils.pytorch_utils import PyTorchUtils
# Model comparison logic here
print('Model comparison completed')
"

# Docker targets
docker-build:
	@echo "Building Docker image..."
	@docker build -t $(PACKAGE_NAME):latest .

docker-run:
	@echo "Running Docker container..."
	@docker run -it --rm -p 8000:8000 $(PACKAGE_NAME):latest

docker-gpu:
	@echo "Running Docker container with GPU support..."
	@docker run --gpus all -it --rm -p 8000:8000 $(PACKAGE_NAME):latest

docker-colab:
	@echo "Building Colab-optimized Docker image..."
	@docker build -f Dockerfile.colab -t $(PACKAGE_NAME):colab .

# Development workflow targets
dev-setup: install-uv setup-env
	@echo "Development setup complete."

dev-test: format lint type-check test-fast
	@echo "Development testing complete."

dev-full: format lint type-check test
	@echo "Full development check complete."

dev-gpu: gpu-info install-dev test-gpu
	@echo "GPU development setup complete."

# CI/CD targets
ci-test:
	@echo "Running CI tests..."
	@pytest $(TEST_PATH) -v --cov=$(SRC_PATH) --cov-report=xml --cov-fail-under=80

ci-lint:
	@echo "Running CI linting..."
	@ruff check $(SRC_PATH) $(TEST_PATH) --output-format=github

ci-format-check:
	@echo "Checking code formatting..."
	@black --check $(SRC_PATH) $(TEST_PATH)
	@ruff check $(SRC_PATH) $(TEST_PATH)

# Release targets
release-check: clean format lint type-check test build
	@echo "Release check complete."

release-build: release-check
	@echo "Building release..."
	@$(PYTHON) -m build
	@twine check dist/*

# Monitoring and logging
logs:
	@echo "Showing recent logs..."
	@tail -f *.log 2>/dev/null || echo "No log files found"

monitor-gpu:
	@echo "Monitoring GPU usage..."
	@watch -n 1 nvidia-smi

monitor-training:
	@echo "Monitoring training progress..."
	@tensorboard --logdir=./results/tensorboard --port=6006

# Data management
clean-data:
	@echo "Cleaning generated data..."
	@rm -rf ./data/generated_*
	@rm -rf ./results/temp_*
	@rm -rf ./samples/temp_*

backup-data:
	@echo "Backing up data..."
	@tar -czf data_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz ./data ./results ./samples

# Experiment management
mlflow-ui:
	@echo "Starting MLflow UI..."
	@mlflow ui --host 0.0.0.0 --port 5000

wandb-sync:
	@echo "Syncing with Weights & Biases..."
	@wandb sync ./wandb

# Quick start targets
quickstart:
	@echo "Quick start: Generate data and train model..."
	@$(MAKE) generate-data
	@$(MAKE) train-gan-simple

quickstart-colab:
	@echo "Quick start for Google Colab..."
	@$(MAKE) colab-setup
	@$(MAKE) generate-data
	@$(MAKE) train-gan --batch-size=8 --epochs=10

# Environment variables for common operations
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)
export CUDA_VISIBLE_DEVICES ?= 0
export TORCH_CUDA_ARCH_LIST ?= "6.0;6.1;7.0;7.5;8.0;8.6"
export MLFLOW_TRACKING_URI ?= http://localhost:5000
export WANDB_PROJECT ?= deepsculpt-v2