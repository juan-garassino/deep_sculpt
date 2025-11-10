# DeepSculpt v2.0 Notebooks

Complete Jupyter notebooks for training and using DeepSculpt models.

## 📓 Available Notebooks

### 1. DeepSculpt_Local_Example.ipynb ⭐ NEW
**Complete local training example with minimal configuration**

- **Purpose**: Quick local testing and development
- **Time**: 5-10 minutes
- **Hardware**: CPU or GPU
- **Configuration**: Small (16³, 20 samples, 3 epochs)

**Workflow**:
1. ✅ Dataset generation (synthetic 3D sculptures)
2. ✅ Data loading and preprocessing
3. ✅ Model creation (Generator + Discriminator)
4. ✅ Training (GAN)
5. ✅ Sample generation
6. ✅ Visualization
7. ✅ Model saving/loading

**Perfect for**:
- Testing the pipeline locally
- Development and debugging
- Understanding the workflow
- Quick experiments

### 2. DeepSculpt_v2_Colab_QuickStart.ipynb
**Simplified Colab version for quick start**

- **Purpose**: Fast Colab training
- **Time**: 10-15 minutes
- **Hardware**: Colab GPU (T4)
- **Configuration**: Small (32³, 20 samples, 3 epochs)

**Features**:
- Colab setup and installation
- Quick GAN training
- Basic visualization
- Sample generation

### 3. DeepSculpt_v2_Colab_Complete.ipynb
**Full-featured production notebook**

- **Purpose**: Complete training pipeline
- **Time**: 4-6 hours
- **Hardware**: Colab GPU (T4 or better)
- **Configuration**: Full (64³, 1000 samples, 100 epochs)

**Features**:
- GAN training (multiple architectures)
- Diffusion model training
- Advanced sampling techniques
- 3D visualization (plotly, open3d)
- Model comparison
- Experiment tracking

## 🚀 Quick Start

### Local Testing
```bash
cd deepsculpt_v2/notebooks
jupyter notebook DeepSculpt_Local_Example.ipynb
# Run all cells → Results in ./local_example_output/
```

### Colab Training
1. Upload `DeepSculpt_Local_Example.ipynb` to Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Add installation cell (see QUICK_START_GUIDE.md)
4. Update configuration for production
5. Run all cells

## 📊 Configuration Comparison

| Feature | Local Example | Colab QuickStart | Colab Complete |
|---------|--------------|------------------|----------------|
| Resolution | 16³ | 32³ | 64³ |
| Samples | 20 | 20 | 1000 |
| Epochs | 3 | 3 | 100 |
| Model | Simple | Simple | Skip/Complex |
| Time | 5-10 min | 10-15 min | 4-6 hours |
| Hardware | CPU/GPU | Colab GPU | Colab GPU |
| Sparse Mode | No | No | Yes |
| Mixed Precision | No | No | Yes |

## 📁 Output Structure

All notebooks produce similar output structure:

```
output_directory/
├── data/                      # Training dataset
│   └── YYYY-MM-DD/
│       ├── pytorch_samples/
│       └── metadata/
├── models/                    # Trained models
│   ├── generator_final.pt
│   ├── discriminator_final.pt
│   └── config.json
├── checkpoints/               # Training checkpoints
│   ├── generator_epoch_*.pt
│   └── discriminator_epoch_*.pt
├── generated_samples/         # Generated sculptures
│   ├── sample_000.pt
│   ├── sample_001.pt
│   └── ...
├── training_metrics.png       # Loss curves
└── generated_samples.png      # Visualizations
```

## 🔧 Requirements

### All Notebooks
- Python 3.8+
- PyTorch 1.12+
- NumPy
- Matplotlib

### Local Example
```bash
pip install torch torchvision numpy matplotlib jupyter
```

### Colab Notebooks
All dependencies installed automatically in the notebook.

## 📚 Documentation

- **QUICK_START_GUIDE.md** - Step-by-step guide for beginners
- **MISSING_FUNCTIONS_ANALYSIS.md** - Technical analysis of implementations
- **Main README** - `../README.md` for full project documentation

## 🎯 Workflow Comparison

### Local Example Workflow
```
Setup (1 min)
  ↓
Generate Data (2 min)
  ↓
Train Model (2-5 min)
  ↓
Generate Samples (1 min)
  ↓
Visualize (1 min)
```

### Colab Complete Workflow
```
Setup & Install (5 min)
  ↓
Generate Data (30 min)
  ↓
Train GAN (2-3 hours)
  ↓
Train Diffusion (2-3 hours)
  ↓
Generate & Compare (30 min)
  ↓
Advanced Visualization (30 min)
```

## 🎨 Customization

### Change Sculpture Style
```python
sculptor_config = {
    'void_dim': 32,
    'edges': (3, 0.2, 0.6),    # More edges
    'planes': (2, 0.3, 0.7),   # More planes
    'pipes': (2, 0.4, 0.8),    # More pipes
}
```

### Change Model Architecture
```python
CONFIG['model_type'] = 'skip'  # Options: simple, complex, skip, autoencoder
```

### Change Training Duration
```python
CONFIG['epochs'] = 100         # More epochs = better quality
CONFIG['batch_size'] = 32      # Larger batch = more stable
```

## 🐛 Troubleshooting

### Import Errors
Make sure you're in the correct directory:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

### Memory Errors
Reduce configuration:
```python
CONFIG['void_dim'] = 16        # Smaller resolution
CONFIG['batch_size'] = 4       # Smaller batches
CONFIG['sparse_mode'] = True   # Enable sparse tensors
```

### Training Issues
Check the QUICK_START_GUIDE.md for detailed troubleshooting.

## 📈 Expected Results

### After Local Training (3 epochs)
- Models saved successfully
- Generated samples show basic structure
- Training losses decrease
- Quick validation of pipeline

### After Colab Training (100 epochs)
- High-quality generated sculptures
- Stable training metrics
- Diverse sample generation
- Production-ready models

## 🔄 Migration Path

1. **Start Local**: Test with `DeepSculpt_Local_Example.ipynb`
2. **Verify**: Check outputs and understand workflow
3. **Migrate**: Upload to Colab with production config
4. **Train**: Full training with larger dataset
5. **Deploy**: Use trained models for generation

## 💡 Tips

### For Best Results
- Start with local testing to verify setup
- Use GPU for training (10-50x faster)
- Enable sparse mode for large resolutions
- Monitor training metrics regularly
- Save checkpoints frequently

### For Experimentation
- Try different model architectures
- Adjust sculptor configuration
- Experiment with hyperparameters
- Compare GAN vs Diffusion models

### For Production
- Use Colab with GPU
- Train for 100+ epochs
- Generate large datasets (1000+ samples)
- Enable mixed precision training
- Save results to Google Drive

## 📞 Support

- **Issues**: Check MISSING_FUNCTIONS_ANALYSIS.md
- **Guide**: Read QUICK_START_GUIDE.md
- **Examples**: See `../examples/` directory
- **API**: Check `../core/` modules

## 🎉 Success Criteria

You'll know it's working when:
- ✅ All cells execute without errors
- ✅ Training losses decrease over time
- ✅ Generated samples show 3D structure
- ✅ Models save and load correctly
- ✅ Visualizations display properly

## 🚀 Next Steps

After completing the local example:
1. Try different configurations
2. Experiment with model architectures
3. Migrate to Colab for full training
4. Explore diffusion models
5. Export sculptures for 3D printing

---

**Ready to create amazing 3D sculptures with AI! 🎨✨**
