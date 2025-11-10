# DeepSculpt Legacy (v1.x)

This folder contains the original TensorFlow-based implementation of DeepSculpt. This is the legacy version (v1.x) that has been preserved for backward compatibility and reference purposes.

## Structure

- `deepSculpt/` - Core legacy modules (TensorFlow-based)
  - `main.py` - Legacy main entry point
  - `models.py` - TensorFlow model architectures
  - `trainer.py` - TensorFlow training infrastructure
  - `collector.py` - Original data collection system
  - `curator.py` - Original data preprocessing
  - `sculptor.py` - Original sculpture generation
  - `shapes.py` - Original shape primitives
  - `utils.py` - Legacy utility functions
  - `visualization.py` - Original visualization system
  - `workflow.py` - Legacy workflow orchestration
  - `logger.py` - Legacy logging system

- `boilerplate/` - API and bot implementations
- `notebooks/` - Jupyter notebooks for experimentation
- `scripts/` - Legacy scripts and utilities
- `summaries/` - Migration and implementation summaries
- `tests/` - Legacy test suite

## Usage

To use the legacy version:

```bash
cd deepsculpt_legacy
python deepSculpt/main.py train --model-type=skip --epochs=100 --data-folder=./data
```

## Migration

For the new PyTorch-based implementation (v2.0), see the `deepsculpt_v2/` folder in the parent directory.

## Maintenance

This legacy codebase is maintained for:
- Backward compatibility
- Reference for migration validation
- Historical preservation of the original implementation

New features and improvements are implemented in DeepSculpt v2.0.