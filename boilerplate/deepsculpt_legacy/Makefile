# DeepSculpt Legacy - TensorFlow Makefile
# Comprehensive build system for legacy TensorFlow operations

.PHONY: help install install-dev test test-fast test-slow lint format type-check clean build docs serve-docs train-model generate-data evaluate-model serve-api run-bot workflow benchmark profile docker-build docker-run

# Default target
help:
	@echo "DeepSculpt Legacy (TensorFlow) - Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-all      Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-fast        Run fast tests only"
	@echo "  test-slow        Run slow/integration tests"
	@echo "  test-gpu         Run GPU-specific tests"
	@echo "  lint             Run linting (flake8)"
	@echo "  format           Format code (black, isort)"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  clean            Clean build artifacts"
	@echo ""
	@echo "Build and Documentation:"
	@echo "  build            Build package"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Machine Learning Operations:"
	@echo "  train-model      Train a TensorFlow GAN model"
	@echo "  generate-data    Generate synthetic training data"
	@echo "  evaluate-model   Evaluate a trained model"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  profile          Profile model performance"
	@echo ""
	@echo "Services:"
	@echo "  serve-api        Start the API server"
	@echo "  run-bot          Start the Telegram bot"
	@echo "  workflow         Run workflow automation"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  check-env        Check environment setup"
	@echo "  setup-env        Setup development environment"

# Variables
PYTHON := python3
PIP := pip3
PACKAGE_NAME := deepsculpt-legacy
TEST_PATH := tests/
SRC_PATH := deepSculpt/
DOCS_PATH := docs/
BUILD_PATH := build/
DIST_PATH := dist/

# Environment setup
check-env:
	@echo "Checking environment setup..."
	@$(PYTHON) --version
	@$(PIP) --version
	@$(PYTHON) -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || echo "TensorFlow not installed"
	@$(PYTHON) -c "import numpy as np; print(f'NumPy: {np.__version__}')" || echo "NumPy not installed"
	@echo "Environment check complete."

setup-env:
	@echo "Setting up development environment..."
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -e ".[dev]"
	@pre-commit install
	@echo "Development environment setup complete."

# Installation targets
install:
	@echo "Installing DeepSculpt Legacy..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e .
	@echo "Installation complete."

install-dev:
	@echo "Installing DeepSculpt Legacy with development dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[dev]"
	@echo "Development installation complete."

install-all:
	@echo "Installing DeepSculpt Legacy with all dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[all]"
	@echo "Full installation complete."

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

test-tensorflow:
	@echo "Running TensorFlow-specific tests..."
	@pytest $(TEST_PATH) -v -m "tensorflow" --cov=$(SRC_PATH)

# Code quality targets
lint:
	@echo "Running linting..."
	@flake8 $(SRC_PATH) $(TEST_PATH)
	@echo "Linting complete."

format:
	@echo "Formatting code..."
	@black $(SRC_PATH) $(TEST_PATH)
	@isort $(SRC_PATH) $(TEST_PATH)
	@echo "Code formatting complete."

type-check:
	@echo "Running type checking..."
	@mypy $(SRC_PATH)
	@echo "Type checking complete."

# Build targets
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_PATH) $(DIST_PATH) *.egg-info
	@rm -rf .pytest_cache .coverage htmlcov
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

# Machine Learning Operations
train-model:
	@echo "Training TensorFlow GAN model..."
	@$(PYTHON) main.py train \
		--model-type=skip \
		--epochs=100 \
		--batch-size=32 \
		--data-folder=./data \
		--output-dir=./results \
		--mlflow \
		--verbose

train-simple:
	@echo "Training simple GAN model..."
	@$(PYTHON) main.py train \
		--model-type=simple \
		--epochs=50 \
		--batch-size=16 \
		--data-folder=./data

train-complex:
	@echo "Training complex GAN model..."
	@$(PYTHON) main.py train \
		--model-type=complex \
		--epochs=200 \
		--batch-size=32 \
		--data-folder=./data \
		--mlflow

generate-data:
	@echo "Generating synthetic training data..."
	@$(PYTHON) main.py generate-data \
		--num-samples=1000 \
		--void-dim=64 \
		--num-shapes=5 \
		--output-dir=./data

generate-data-large:
	@echo "Generating large dataset..."
	@$(PYTHON) main.py generate-data \
		--num-samples=10000 \
		--void-dim=64 \
		--num-shapes=7 \
		--output-dir=./data/large

evaluate-model:
	@echo "Evaluating trained model..."
	@$(PYTHON) main.py evaluate \
		--checkpoint=./results/generator_final \
		--num-samples=20 \
		--visualize \
		--output-dir=./evaluation

benchmark:
	@echo "Running performance benchmarks..."
	@$(PYTHON) -m pytest tests/benchmarks/ -v --benchmark-only

profile:
	@echo "Profiling model performance..."
	@$(PYTHON) -m cProfile -o profile_results.prof main.py train --epochs=1 --batch-size=8
	@$(PYTHON) -c "import pstats; pstats.Stats('profile_results.prof').sort_stats('cumulative').print_stats(20)"

# Service targets
serve-api:
	@echo "Starting API server..."
	@$(PYTHON) main.py serve-api --host=0.0.0.0 --port=8000 --reload

serve-api-prod:
	@echo "Starting API server in production mode..."
	@$(PYTHON) main.py serve-api --host=0.0.0.0 --port=8000

run-bot:
	@echo "Starting Telegram bot..."
	@$(PYTHON) main.py run-bot --token=$(TELEGRAM_TOKEN)

workflow:
	@echo "Running workflow automation..."
	@$(PYTHON) main.py workflow --mode=development

workflow-prod:
	@echo "Running workflow in production mode..."
	@$(PYTHON) main.py workflow --mode=production

all-services:
	@echo "Starting all services..."
	@$(PYTHON) main.py all --mode=development --bot-token=$(TELEGRAM_TOKEN)

# Docker targets
docker-build:
	@echo "Building Docker image..."
	@docker build -t $(PACKAGE_NAME):latest .

docker-run:
	@echo "Running Docker container..."
	@docker run -it --rm -p 8000:8000 $(PACKAGE_NAME):latest

docker-run-gpu:
	@echo "Running Docker container with GPU support..."
	@docker run --gpus all -it --rm -p 8000:8000 $(PACKAGE_NAME):latest

# Development workflow targets
dev-setup: install-dev setup-env
	@echo "Development setup complete."

dev-test: format lint type-check test-fast
	@echo "Development testing complete."

dev-full: format lint type-check test
	@echo "Full development check complete."

# CI/CD targets
ci-test:
	@echo "Running CI tests..."
	@pytest $(TEST_PATH) -v --cov=$(SRC_PATH) --cov-report=xml --cov-fail-under=80

ci-lint:
	@echo "Running CI linting..."
	@flake8 $(SRC_PATH) $(TEST_PATH) --count --select=E9,F63,F7,F82 --show-source --statistics
	@flake8 $(SRC_PATH) $(TEST_PATH) --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

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

monitor:
	@echo "Starting system monitoring..."
	@$(PYTHON) -c "
import psutil
import time
while True:
    print(f'CPU: {psutil.cpu_percent()}% | Memory: {psutil.virtual_memory().percent}%')
    time.sleep(5)
"

# Data management
clean-data:
	@echo "Cleaning generated data..."
	@rm -rf ./data/generated_*
	@rm -rf ./results/temp_*

backup-data:
	@echo "Backing up data..."
	@tar -czf data_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz ./data ./results

# Environment variables for common operations
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)
export CUDA_VISIBLE_DEVICES ?= 0
export TF_CPP_MIN_LOG_LEVEL ?= 2
export MLFLOW_TRACKING_URI ?= http://localhost:5000