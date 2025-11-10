#!/usr/bin/env python3
"""
Script to create all Colab notebook variations.
Run: python scripts/create_all_notebooks.py
"""

import json
from pathlib import Path


def create_base_cells():
    """Common cells for all notebooks."""
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Setup - Clone Repository"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone repository (update YOUR_USERNAME)\n",
                "!git clone https://github.com/YOUR_USERNAME/deepSculpt.git\n",
                "%cd deepSculpt/deepsculpt_v2"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Imports"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "from pathlib import Path\n",
                "import time\n",
                "import json\n",
                "\n",
                "sys.path.insert(0, str(Path.cwd()))\n",
                "\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.multiprocessing as mp\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "\n",
                "# Fix for Colab multiprocessing\n",
                "mp.set_start_method('spawn', force=True)\n",
                "\n",
                "print(f\"PyTorch: {torch.__version__}\")\n",
                "print(f\"CUDA: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
            ]
        }
    ]


def create_dataset_class_cell():
    """Dataset class cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class SimpleDataset(Dataset):\n",
            "    def __init__(self, file_paths, device='cpu'):\n",
            "        self.file_paths = file_paths\n",
            "        self.device = device\n",
            "    \n",
            "    def __len__(self):\n",
            "        return len(self.file_paths)\n",
            "    \n",
            "    def __getitem__(self, idx):\n",
            "        structure = torch.load(self.file_paths[idx])\n",
            "        if structure.dim() == 3:\n",
            "            structure = structure.unsqueeze(0)\n",
            "        structure = structure.float()\n",
            "        if structure.max() > 1.0:\n",
            "            structure = structure / 255.0\n",
            "        return structure.to(self.device)"
        ]
    }


def create_gan_notebook(color_mode, output_name):
    """Create GAN training notebook."""
    color_str = "Color" if color_mode == 1 else "Monochrome"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# DeepSculpt v2.0 - GAN Training ({color_str})\n",
                "\n",
                f"Train a GAN model to generate {color_str.lower()} 3D sculptures.\n",
                "\n",
                "**Setup:** Enable GPU (Runtime → Change runtime type → GPU)"
            ]
        }
    ]
    
    cells.extend(create_base_cells())
    
    # Import cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from core.data.generation.pytorch_collector import PyTorchCollector\n",
            "from core.models.model_factory import PyTorchModelFactory\n",
            "from core.training.pytorch_trainer import GANTrainer, TrainingConfig\n",
            "\n",
            "print('✅ Modules loaded')"
        ]
    })

    
    # Config cell
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## 3. Configuration ({color_str})"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "CONFIG = {\n",
                "    'void_dim': 32,\n",
                "    'num_samples': 100,\n",
                "    'model_type': 'simple',\n",
                "    'noise_dim': 100,\n",
                f"    'color_mode': {color_mode},  # {color_str}\n",
                "    'epochs': 20,\n",
                "    'batch_size': 16,\n",
                "    'learning_rate': 0.0002,\n",
                "    'beta1': 0.5,\n",
                "    'beta2': 0.999,\n",
                "    'num_eval_samples': 5,\n",
                "    'device': 'cuda' if torch.cuda.is_available() else 'cpu',\n",
                "    'sparse_mode': False,\n",
                "    'mixed_precision': True,\n",
                "    'output_dir': '/content/output',\n",
                "}\n",
                "\n",
                "for k, v in CONFIG.items():\n",
                "    print(f'{k}: {v}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Dataset Class"]
        },
        create_dataset_class_cell(),
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Generate Dataset"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "output_dir = Path(CONFIG['output_dir'])\n",
                "data_dir = output_dir / 'data'\n",
                "data_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "sculptor_config = {\n",
                "    'void_dim': CONFIG['void_dim'],\n",
                "    'edges': (1, 0.3, 0.5),\n",
                "    'planes': (1, 0.3, 0.5),\n",
                "    'pipes': (1, 0.3, 0.5),\n",
                "    'grid': (1, 4),\n",
                "    'step': 1,\n",
                "}\n",
                "\n",
                "collector = PyTorchCollector(\n",
                "    sculptor_config=sculptor_config,\n",
                "    output_format='pytorch',\n",
                "    base_dir=str(data_dir),\n",
                "    device=CONFIG['device'],\n",
                "    sparse_mode=CONFIG['sparse_mode'],\n",
                "    verbose=True,\n",
                ")\n",
                "\n",
                "print(f\"🎨 Generating {CONFIG['num_samples']} samples...\")\n",
                "start = time.time()\n",
                "\n",
                "sample_paths = collector.create_collection(\n",
                "    num_samples=CONFIG['num_samples'],\n",
                "    batch_size=CONFIG['batch_size'],\n",
                "    dynamic_batching=False,\n",
                ")\n",
                "\n",
                "print(f'✅ Generated in {time.time()-start:.1f}s')"
            ]
        }
    ])

    
    # DataLoader, Models, Training cells
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. DataLoader"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset = SimpleDataset(sample_paths, device=CONFIG['device'])\n",
                "dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)\n",
                "print(f'Dataset: {len(dataset)} samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Create Models"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "factory = PyTorchModelFactory(device=CONFIG['device'])\n",
                "\n",
                "generator = factory.create_gan_generator(\n",
                "    model_type=CONFIG['model_type'],\n",
                "    void_dim=CONFIG['void_dim'],\n",
                "    noise_dim=CONFIG['noise_dim'],\n",
                "    color_mode=CONFIG['color_mode'],\n",
                "    sparse=CONFIG['sparse_mode'],\n",
                ")\n",
                "\n",
                "discriminator = factory.create_gan_discriminator(\n",
                "    model_type=CONFIG['model_type'],\n",
                "    void_dim=CONFIG['void_dim'],\n",
                "    color_mode=CONFIG['color_mode'],\n",
                "    sparse=CONFIG['sparse_mode'],\n",
                ")\n",
                "\n",
                "print('✅ Models created')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Training"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "log_dir = output_dir / 'logs'\n",
                "log_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "training_config = TrainingConfig(\n",
                "    epochs=CONFIG['epochs'],\n",
                "    batch_size=CONFIG['batch_size'],\n",
                "    learning_rate=CONFIG['learning_rate'],\n",
                "    mixed_precision=CONFIG['mixed_precision'],\n",
                "    gradient_clip=1.0,\n",
                "    checkpoint_freq=5,\n",
                "    log_dir=str(log_dir),\n",
                "    use_tensorboard=False,\n",
                "    use_wandb=False,\n",
                "    use_mlflow=False,\n",
                ")\n",
                "\n",
                "gen_optimizer = torch.optim.Adam(generator.parameters(), lr=CONFIG['learning_rate'], betas=(CONFIG['beta1'], CONFIG['beta2']))\n",
                "disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=CONFIG['learning_rate'], betas=(CONFIG['beta1'], CONFIG['beta2']))\n",
                "\n",
                "trainer = GANTrainer(\n",
                "    generator=generator,\n",
                "    discriminator=discriminator,\n",
                "    gen_optimizer=gen_optimizer,\n",
                "    disc_optimizer=disc_optimizer,\n",
                "    config=training_config,\n",
                "    device=CONFIG['device'],\n",
                "    noise_dim=CONFIG['noise_dim'],\n",
                ")\n",
                "\n",
                "print('✅ Trainer ready')"
            ]
        }
    ])

    
    # Training loop and visualization
    cells.extend([
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "checkpoint_dir = output_dir / 'checkpoints'\n",
                "checkpoint_dir.mkdir(exist_ok=True)\n",
                "\n",
                "print(f\"🚀 Training {CONFIG['epochs']} epochs...\\n\")\n",
                "start = time.time()\n",
                "\n",
                "all_metrics = {'gen_loss': [], 'disc_loss': [], 'disc_real_acc': [], 'disc_fake_acc': []}\n",
                "\n",
                "for epoch in range(CONFIG['epochs']):\n",
                "    print(f\"Epoch {epoch+1}/{CONFIG['epochs']}\")\n",
                "    epoch_metrics = trainer.train_epoch(dataloader)\n",
                "    \n",
                "    for key in all_metrics.keys():\n",
                "        if key in epoch_metrics:\n",
                "            all_metrics[key].append(epoch_metrics[key])\n",
                "    \n",
                "    print(f\"  Gen: {epoch_metrics.get('gen_loss', 0):.4f}, Disc: {epoch_metrics.get('disc_loss', 0):.4f}\")\n",
                "    \n",
                "    if (epoch + 1) % training_config.checkpoint_freq == 0:\n",
                "        trainer.save_checkpoint(str(checkpoint_dir / f'checkpoint_{epoch+1}.pth'), epoch+1, epoch_metrics, False)\n",
                "        print('  💾 Saved')\n",
                "\n",
                "print(f'\\n✅ Done in {time.time()-start:.1f}s')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Plot Metrics"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "axes[0].plot(all_metrics['gen_loss'], label='Generator', marker='o')\n",
                "axes[0].plot(all_metrics['disc_loss'], label='Discriminator', marker='s')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].set_ylabel('Loss')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True, alpha=0.3)\n",
                "\n",
                "axes[1].plot(all_metrics['disc_real_acc'], label='Real', marker='o')\n",
                "axes[1].plot(all_metrics['disc_fake_acc'], label='Fake', marker='s')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_ylabel('Accuracy')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(output_dir / 'metrics.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Save Models"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models_dir = output_dir / 'models'\n",
                "models_dir.mkdir(exist_ok=True)\n",
                "\n",
                "torch.save(generator.state_dict(), models_dir / 'generator.pt')\n",
                "torch.save(discriminator.state_dict(), models_dir / 'discriminator.pt')\n",
                "\n",
                "with open(models_dir / 'config.json', 'w') as f:\n",
                "    json.dump({'void_dim': CONFIG['void_dim'], 'noise_dim': CONFIG['noise_dim'], 'color_mode': CONFIG['color_mode']}, f)\n",
                "\n",
                "print('✅ Saved')"
            ]
        }
    ])

    
    # Generation and visualization
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 11. Generate Samples"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "samples_dir = output_dir / 'samples'\n",
                "samples_dir.mkdir(exist_ok=True)\n",
                "\n",
                "generator.eval()\n",
                "generated = []\n",
                "\n",
                "with torch.no_grad():\n",
                "    for i in range(CONFIG['num_eval_samples']):\n",
                "        noise = torch.randn(1, CONFIG['noise_dim'], device=CONFIG['device'])\n",
                "        sample = generator(noise)\n",
                "        generated.append(sample.cpu())\n",
                "        torch.save(sample.cpu(), samples_dir / f'sample_{i:03d}.pt')\n",
                "\n",
                "print(f'✅ Generated {len(generated)} samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 12. Visualize"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, len(generated), figsize=(5*len(generated), 5))\n",
                "if len(generated) == 1:\n",
                "    axes = [axes]\n",
                "\n",
                "for i, sample in enumerate(generated):\n",
                "    s = sample.squeeze().numpy()\n",
                "    if s.ndim == 4:\n",
                "        s = s[0]\n",
                "    mid = s.shape[0] // 2\n",
                "    axes[i].imshow(s[mid], cmap='viridis')\n",
                "    axes[i].set_title(f'Sample {i+1}')\n",
                "    axes[i].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(output_dir / 'samples.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 13. Download Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!zip -r /content/results.zip {CONFIG['output_dir']}\n",
                "from google.colab import files\n",
                "files.download('/content/results.zip')"
            ]
        }
    ])
    
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook



def create_diffusion_notebook(color_mode, output_name):
    """Create Diffusion training notebook."""
    color_str = "Color" if color_mode == 1 else "Monochrome"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# DeepSculpt v2.0 - Diffusion Training ({color_str})\n",
                "\n",
                f"Train a Diffusion model to generate {color_str.lower()} 3D sculptures.\n",
                "\n",
                "**Setup:** Enable GPU (Runtime → Change runtime type → GPU)"
            ]
        }
    ]
    
    cells.extend(create_base_cells())
    
    # Import cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from core.data.generation.pytorch_collector import PyTorchCollector\n",
            "from core.models.model_factory import PyTorchModelFactory\n",
            "from core.training.pytorch_trainer import DiffusionTrainer, TrainingConfig\n",
            "\n",
            "print('✅ Modules loaded')"
        ]
    })
    
    # Config
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## 3. Configuration ({color_str})"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "CONFIG = {\n",
                "    'void_dim': 32,\n",
                "    'num_samples': 100,\n",
                "    'model_type': 'unet',\n",
                f"    'color_mode': {color_mode},  # {color_str}\n",
                "    'timesteps': 1000,\n",
                "    'epochs': 30,\n",
                "    'batch_size': 8,\n",
                "    'learning_rate': 0.0001,\n",
                "    'num_eval_samples': 5,\n",
                "    'device': 'cuda' if torch.cuda.is_available() else 'cpu',\n",
                "    'sparse_mode': False,\n",
                "    'mixed_precision': True,\n",
                "    'output_dir': '/content/output',\n",
                "}\n",
                "\n",
                "for k, v in CONFIG.items():\n",
                "    print(f'{k}: {v}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Dataset Class"]
        },
        create_dataset_class_cell()
    ])

    
    # Dataset generation (same as GAN)
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Generate Dataset"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "output_dir = Path(CONFIG['output_dir'])\n",
                "data_dir = output_dir / 'data'\n",
                "data_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "sculptor_config = {\n",
                "    'void_dim': CONFIG['void_dim'],\n",
                "    'edges': (1, 0.3, 0.5),\n",
                "    'planes': (1, 0.3, 0.5),\n",
                "    'pipes': (1, 0.3, 0.5),\n",
                "    'grid': (1, 4),\n",
                "    'step': 1,\n",
                "}\n",
                "\n",
                "collector = PyTorchCollector(\n",
                "    sculptor_config=sculptor_config,\n",
                "    output_format='pytorch',\n",
                "    base_dir=str(data_dir),\n",
                "    device=CONFIG['device'],\n",
                "    sparse_mode=CONFIG['sparse_mode'],\n",
                "    verbose=True,\n",
                ")\n",
                "\n",
                "print(f\"🎨 Generating {CONFIG['num_samples']} samples...\")\n",
                "start = time.time()\n",
                "\n",
                "sample_paths = collector.create_collection(\n",
                "    num_samples=CONFIG['num_samples'],\n",
                "    batch_size=CONFIG['batch_size'],\n",
                "    dynamic_batching=False,\n",
                ")\n",
                "\n",
                "print(f'✅ Generated in {time.time()-start:.1f}s')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. DataLoader"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset = SimpleDataset(sample_paths, device=CONFIG['device'])\n",
                "dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)\n",
                "print(f'Dataset: {len(dataset)} samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Create Diffusion Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "factory = PyTorchModelFactory(device=CONFIG['device'])\n",
                "\n",
                "model = factory.create_diffusion_model(\n",
                "    model_type=CONFIG['model_type'],\n",
                "    void_dim=CONFIG['void_dim'],\n",
                "    color_mode=CONFIG['color_mode'],\n",
                "    timesteps=CONFIG['timesteps'],\n",
                "    sparse=CONFIG['sparse_mode'],\n",
                ")\n",
                "\n",
                "print('✅ Model created')"
            ]
        }
    ])

    
    # Training
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Training"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "log_dir = output_dir / 'logs'\n",
                "log_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "training_config = TrainingConfig(\n",
                "    epochs=CONFIG['epochs'],\n",
                "    batch_size=CONFIG['batch_size'],\n",
                "    learning_rate=CONFIG['learning_rate'],\n",
                "    mixed_precision=CONFIG['mixed_precision'],\n",
                "    gradient_clip=1.0,\n",
                "    checkpoint_freq=10,\n",
                "    log_dir=str(log_dir),\n",
                "    use_tensorboard=False,\n",
                "    use_wandb=False,\n",
                "    use_mlflow=False,\n",
                ")\n",
                "\n",
                "optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])\n",
                "\n",
                "trainer = DiffusionTrainer(\n",
                "    model=model,\n",
                "    optimizer=optimizer,\n",
                "    config=training_config,\n",
                "    device=CONFIG['device'],\n",
                "    timesteps=CONFIG['timesteps'],\n",
                ")\n",
                "\n",
                "print('✅ Trainer ready')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "checkpoint_dir = output_dir / 'checkpoints'\n",
                "checkpoint_dir.mkdir(exist_ok=True)\n",
                "\n",
                "print(f\"🚀 Training {CONFIG['epochs']} epochs...\\n\")\n",
                "start = time.time()\n",
                "\n",
                "all_metrics = {'loss': []}\n",
                "\n",
                "for epoch in range(CONFIG['epochs']):\n",
                "    print(f\"Epoch {epoch+1}/{CONFIG['epochs']}\")\n",
                "    epoch_metrics = trainer.train_epoch(dataloader)\n",
                "    \n",
                "    if 'loss' in epoch_metrics:\n",
                "        all_metrics['loss'].append(epoch_metrics['loss'])\n",
                "    \n",
                "    print(f\"  Loss: {epoch_metrics.get('loss', 0):.4f}\")\n",
                "    \n",
                "    if (epoch + 1) % training_config.checkpoint_freq == 0:\n",
                "        trainer.save_checkpoint(str(checkpoint_dir / f'checkpoint_{epoch+1}.pth'), epoch+1, epoch_metrics, False)\n",
                "        print('  💾 Saved')\n",
                "\n",
                "print(f'\\n✅ Done in {time.time()-start:.1f}s')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Plot Loss"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "plt.plot(all_metrics['loss'], marker='o')\n",
                "plt.xlabel('Epoch')\n",
                "plt.ylabel('Loss')\n",
                "plt.title('Diffusion Training Loss')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.savefig(output_dir / 'loss.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Save Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models_dir = output_dir / 'models'\n",
                "models_dir.mkdir(exist_ok=True)\n",
                "\n",
                "torch.save(model.state_dict(), models_dir / 'diffusion_model.pt')\n",
                "\n",
                "with open(models_dir / 'config.json', 'w') as f:\n",
                "    json.dump({'void_dim': CONFIG['void_dim'], 'timesteps': CONFIG['timesteps'], 'color_mode': CONFIG['color_mode']}, f)\n",
                "\n",
                "print('✅ Saved')"
            ]
        }
    ])

    
    # Generation
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 11. Generate Samples"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "samples_dir = output_dir / 'samples'\n",
                "samples_dir.mkdir(exist_ok=True)\n",
                "\n",
                "model.eval()\n",
                "generated = []\n",
                "\n",
                "print('🎨 Generating samples (this takes time)...')\n",
                "\n",
                "with torch.no_grad():\n",
                "    for i in range(CONFIG['num_eval_samples']):\n",
                "        # Start from random noise\n",
                "        shape = (1, 1 if CONFIG['color_mode'] == 0 else 6, CONFIG['void_dim'], CONFIG['void_dim'], CONFIG['void_dim'])\n",
                "        sample = trainer.sample(shape)\n",
                "        generated.append(sample.cpu())\n",
                "        torch.save(sample.cpu(), samples_dir / f'sample_{i:03d}.pt')\n",
                "        print(f'  Sample {i+1}/{CONFIG[\"num_eval_samples\"]}')\n",
                "\n",
                "print(f'✅ Generated {len(generated)} samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 12. Visualize"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, len(generated), figsize=(5*len(generated), 5))\n",
                "if len(generated) == 1:\n",
                "    axes = [axes]\n",
                "\n",
                "for i, sample in enumerate(generated):\n",
                "    s = sample.squeeze().numpy()\n",
                "    if s.ndim == 4:\n",
                "        s = s[0]\n",
                "    mid = s.shape[0] // 2\n",
                "    axes[i].imshow(s[mid], cmap='viridis')\n",
                "    axes[i].set_title(f'Sample {i+1}')\n",
                "    axes[i].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(output_dir / 'samples.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 13. Download Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!zip -r /content/results.zip {CONFIG['output_dir']}\n",
                "from google.colab import files\n",
                "files.download('/content/results.zip')"
            ]
        }
    ])
    
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook


# Main execution
if __name__ == "__main__":
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    notebooks = [
        (0, "DeepSculpt_GAN_Monochrome.ipynb", create_gan_notebook),
        (1, "DeepSculpt_GAN_Color.ipynb", create_gan_notebook),
        (0, "DeepSculpt_Diffusion_Monochrome.ipynb", create_diffusion_notebook),
        (1, "DeepSculpt_Diffusion_Color.ipynb", create_diffusion_notebook),
    ]
    
    for color_mode, filename, create_func in notebooks:
        notebook = create_func(color_mode, filename)
        output_path = notebooks_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"✅ Created: {filename}")
    
    print(f"\n🎉 All notebooks created in {notebooks_dir}")
