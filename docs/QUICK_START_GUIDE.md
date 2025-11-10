# DeepSculpt v2.0 - Quick Start Guide

## 🚀 Getting Started

### Local Testing (5-10 minutes)

1. **Navigate to the notebook directory**:
```bash
cd deepSculpt/deepsculpt_v2/notebooks
```

2. **Open the notebook**:
```bash
jupyter notebook DeepSculpt_Local_Example.ipynb
```

3. **Run all cells** - The notebook will:
   - Generate 20 small 3D sculptures (16³ resolution)
   - Train a simple GAN for 3 epochs
   - Generate 3 new samples
   - Save everything to `./local_example_output/`

### Expected Output Structure
```
local_example_output/
├── data/                      # Training dataset
│   └── 2024-XX-XX/
│       └── pytorch_samples/
├── models/                    # Trained models
│   ├── generator_final.pt
│   ├── discriminator_final.pt
│   └── config.json
├── checkpoints/               # Training checkpoints
├── generated_samples/         # Generated sculptures
│   ├── sample_000.pt
│   ├── sample_001.pt
│   └── sample_002.pt
├── training_metrics.png       # Loss curves
└── generated_samples.png      # Visualizations
```

## 📊 Configuration Presets

### Local Testing (Default)
```python
CONFIG = {
    'void_dim': 16,           # 16³ = 4,096 voxels
    'num_samples': 20,        # Small dataset
    'epochs': 3,              # Quick training
    'batch_size': 4,
    'model_type': 'simple',
}
# Time: ~5-10 minutes on CPU, ~2-3 minutes on GPU
```

### Medium Testing
```python
CONFIG = {
    'void_dim': 32,           # 32³ = 32,768 voxels
    'num_samples': 100,
    'epochs': 10,
    'batch_size': 8,
    'model_type': 'complex',
}
# Time: ~30-45 minutes on GPU
```

### Production (Colab)
```python
CONFIG = {
    'void_dim': 64,           # 64³ = 262,144 voxels
    'num_samples': 1000,
    'epochs': 100,
    'batch_size': 32,
    'model_type': 'skip',
    'sparse_mode': True,
    'mixed_precision': True,
}
# Time: ~4-6 hours on Colab T4 GPU
```

## 🔧 Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'core'`

**Solution**: Make sure you're running from the correct directory:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

### Memory Errors
**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `void_dim`: 64 → 32 → 16
2. Reduce `batch_size`: 32 → 16 → 8 → 4
3. Enable sparse mode: `CONFIG['sparse_mode'] = True`
4. Clear cache: `torch.cuda.empty_cache()`

### Training Instability
**Problem**: Loss becomes NaN or training diverges

**Solutions**:
1. Reduce learning rate: `0.0002` → `0.0001`
2. Increase batch size for stability
3. Try different model type: `'simple'` instead of `'complex'`
4. Check data quality (visualize samples)

## 📈 Monitoring Training

### Key Metrics to Watch

1. **Generator Loss**: Should decrease over time
   - Too high (>5): Generator not learning
   - Too low (<0.1): Possible mode collapse

2. **Discriminator Loss**: Should stabilize around 0.5-1.0
   - Too high (>3): Discriminator failing
   - Too low (<0.1): Discriminator too strong

3. **Discriminator Accuracy**:
   - Real accuracy: Should be 70-90%
   - Fake accuracy: Should be 30-70%
   - Both at 50%: Good balance

### Good Training Signs
- ✅ Losses decrease initially then stabilize
- ✅ Generated samples show structure
- ✅ Discriminator accuracy balanced
- ✅ No NaN values

### Bad Training Signs
- ❌ Losses explode to infinity
- ❌ Losses collapse to zero
- ❌ Generated samples are all zeros/ones
- ❌ Discriminator accuracy at 100% or 0%

## 🎨 Customizing Sculptures

### Modify Sculptor Configuration

```python
sculptor_config = {
    'void_dim': 32,
    'edges': (3, 0.2, 0.6),    # More edges
    'planes': (2, 0.3, 0.7),   # More planes
    'pipes': (2, 0.4, 0.8),    # More pipes
    'grid': (1, 4),            # Grid enabled
    'step': 1,
}
```

### Shape Parameters
- **First number**: Count (how many of this shape)
- **Second number**: Min ratio (minimum size as fraction of void_dim)
- **Third number**: Max ratio (maximum size as fraction of void_dim)

### Examples

**Minimal sculptures** (sparse):
```python
'edges': (1, 0.1, 0.3),
'planes': (0, 0, 0),
'pipes': (1, 0.2, 0.4),
```

**Complex sculptures** (dense):
```python
'edges': (5, 0.3, 0.7),
'planes': (3, 0.4, 0.8),
'pipes': (3, 0.5, 0.9),
```

## 🔄 Migrating to Colab

### Step 1: Upload Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Select `DeepSculpt_Local_Example.ipynb`

### Step 2: Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4)
3. Save

### Step 3: Add Installation Cell (at top)
```python
# Install DeepSculpt
!git clone https://github.com/your-org/deepsculpt.git
%cd deepsculpt/deepsculpt_v2

# Install dependencies
!pip install torch torchvision torchaudio
!pip install -r requirements.txt

# Verify
import torch
print(f"CUDA: {torch.cuda.is_available()}")
```

### Step 4: Update Configuration
```python
# Uncomment the Colab configuration section
CONFIG.update({
    'void_dim': 64,
    'num_samples': 1000,
    'epochs': 100,
    'batch_size': 32,
    'model_type': 'skip',
    'sparse_mode': True,
    'mixed_precision': True,
})
```

### Step 5: Run All Cells
- Runtime → Run all
- Monitor progress (will take several hours)

## 💾 Saving Results

### Local
Results automatically saved to `./local_example_output/`

### Colab (to Google Drive)
Add this cell after imports:
```python
from google.colab import drive
drive.mount('/content/drive')

# Update output directory
CONFIG['output_dir'] = '/content/drive/MyDrive/deepsculpt_results'
```

## 🎯 Next Steps

### After Local Testing
1. ✅ Verify notebook runs without errors
2. ✅ Check generated samples look reasonable
3. ✅ Review training metrics
4. → Migrate to Colab for full training

### After Colab Training
1. ✅ Download trained models
2. ✅ Generate more samples
3. ✅ Try different model architectures
4. → Experiment with diffusion models
5. → Export for 3D printing

### Advanced Features
- **Diffusion Models**: See `DeepSculpt_v2_Colab_Complete.ipynb`
- **Custom Architectures**: Modify model_factory.py
- **Advanced Training**: Add learning rate scheduling, gradient clipping
- **Evaluation Metrics**: Add FID, IS scores
- **3D Visualization**: Use plotly or open3d backends

## 📚 Additional Resources

### Documentation
- Main README: `deepsculpt_v2/README.md`
- API Docs: `deepsculpt_v2/docs/`
- Examples: `deepsculpt_v2/examples/`

### Other Notebooks
- `DeepSculpt_v2_Colab_QuickStart.ipynb` - Simplified Colab version
- `DeepSculpt_v2_Colab_Complete.ipynb` - Full features + diffusion

### Command Line Interface
```bash
# Generate data
python main.py generate-data --num-samples=100 --void-dim=32

# Train GAN
python main.py train-gan --model-type=skip --epochs=50

# Train Diffusion
python main.py train-diffusion --epochs=50 --timesteps=1000

# Generate samples
python main.py sample-gan --checkpoint=./models/generator.pt --num-samples=10
```

## ❓ FAQ

**Q: How long does training take?**
A: Local (3 epochs): 5-10 min. Colab (100 epochs): 4-6 hours.

**Q: Can I use CPU only?**
A: Yes, but it's much slower. Reduce void_dim to 16 and num_samples to 20.

**Q: What if I get CUDA out of memory?**
A: Reduce batch_size, void_dim, or enable sparse_mode.

**Q: How do I know if training is working?**
A: Check that losses decrease and generated samples show structure.

**Q: Can I resume training?**
A: Yes, load the checkpoint and continue training.

**Q: How do I export for 3D printing?**
A: Use the visualization module to export to STL format.

## 🐛 Known Issues

1. **Import paths**: May need adjustment depending on directory structure
2. **Logger module**: Ensure logger.py exists in core/utils/
3. **Sparse tensors**: May have compatibility issues on some PyTorch versions
4. **Visualization**: plotly may not work in all environments (use matplotlib fallback)

## 📞 Support

- GitHub Issues: [github.com/your-org/deepsculpt/issues](https://github.com/your-org/deepsculpt/issues)
- Documentation: [docs.deepsculpt.ai](https://docs.deepsculpt.ai)
- Discord: [discord.gg/deepsculpt](https://discord.gg/deepsculpt)

---

**Happy Sculpting! 🎨✨**
