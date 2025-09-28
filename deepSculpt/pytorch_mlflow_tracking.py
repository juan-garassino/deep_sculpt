"""
Enhanced MLflow Tracking for PyTorch Models in DeepSculpt

This module provides comprehensive MLflow integration specifically designed for PyTorch models,
including advanced metrics tracking, model comparison utilities, and diffusion model support.

Key Features:
- PyTorch-specific model logging and artifact management
- Advanced metrics tracking for GAN and diffusion models
- Model comparison between TensorFlow and PyTorch versions
- GPU memory and performance monitoring
- Sparse tensor metrics and optimization tracking
- Custom visualizations for 3D generation models
- Experiment organization and tagging

Dependencies:
- mlflow: For experiment tracking and model registry
- torch: For PyTorch model handling
- matplotlib: For visualization generation
- numpy: For numerical operations
- plotly: For interactive 3D visualizations

Used by:
- pytorch_workflow.py: For experiment tracking in workflows
- pytorch_trainer.py: For training progress tracking
- Model evaluation scripts: For performance comparison
"""

import os
import json
import time
import tempfile
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available, experiment tracking disabled")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available, interactive visualizations disabled")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PyTorchMLflowTracker:
    """
    Enhanced MLflow tracker specifically designed for PyTorch models in DeepSculpt.
    
    This class provides comprehensive experiment tracking capabilities including:
    - Model architecture logging and visualization
    - Training metrics with PyTorch-specific information
    - GPU memory and performance monitoring
    - Model comparison utilities
    - Custom artifact logging for 3D generation models
    """
    
    def __init__(
        self,
        experiment_name: str = "deepSculpt_pytorch",
        tracking_uri: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        enable_wandb: bool = False
    ):
        """
        Initialize the PyTorch MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            model_name: Name for model registry
            tags: Additional tags for the experiment
            enable_wandb: Whether to enable Weights & Biases integration
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for experiment tracking")
        
        self.experiment_name = experiment_name
        self.model_name = model_name or f"{experiment_name}_model"
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Initialize Weights & Biases if enabled
        if self.enable_wandb:
            wandb.init(project=experiment_name, tags=list(tags.values()) if tags else None)
        
        # Default tags
        self.default_tags = {
            "framework": "pytorch",
            "project": "deepSculpt",
            "timestamp": datetime.now().isoformat(),
            **(tags or {})
        }
        
        # Tracking state
        self.current_run = None
        self.run_metrics = {}
        self.model_artifacts = {}
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Additional tags for this run
            
        Returns:
            Run ID
        """
        combined_tags = {**self.default_tags, **(tags or {})}
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=combined_tags)
        self.run_metrics = {}
        self.model_artifacts = {}
        
        return self.current_run.info.run_id
    
    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None
            
        if self.enable_wandb:
            wandb.finish()
    
    def log_model_architecture(
        self,
        model: nn.Module,
        model_type: str = "generator",
        input_shape: Optional[Tuple[int, ...]] = None,
        save_summary: bool = True
    ):
        """
        Log PyTorch model architecture and parameters.
        
        Args:
            model: PyTorch model to log
            model_type: Type of model (generator, discriminator, diffusion)
            input_shape: Input shape for model summary
            save_summary: Whether to save detailed model summary
        """
        if not self.current_run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Basic model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            f"{model_type}_total_parameters": total_params,
            f"{model_type}_trainable_parameters": trainable_params,
            f"{model_type}_model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            f"{model_type}_architecture": model.__class__.__name__,
        }
        
        # Log as parameters
        mlflow.log_params(model_info)
        
        # Log to Weights & Biases if enabled
        if self.enable_wandb:
            wandb.log(model_info)
        
        # Device information
        if torch.cuda.is_available():
            device_info = {
                f"{model_type}_device": str(next(model.parameters()).device),
                f"{model_type}_cuda_available": True,
                f"{model_type}_gpu_count": torch.cuda.device_count(),
            }
            mlflow.log_params(device_info)
            
            if self.enable_wandb:
                wandb.log(device_info)
        
        # Save detailed model summary if requested
        if save_summary:
            self._save_model_summary(model, model_type, input_shape)
        
        # Store model reference for later logging
        self.model_artifacts[model_type] = model
    
    def _save_model_summary(
        self,
        model: nn.Module,
        model_type: str,
        input_shape: Optional[Tuple[int, ...]] = None
    ):
        """Save detailed model summary as artifact."""
        try:
            # Create model summary
            summary_lines = []
            summary_lines.append(f"{model_type.title()} Model Summary")
            summary_lines.append("=" * 50)
            summary_lines.append(f"Architecture: {model.__class__.__name__}")
            summary_lines.append(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
            summary_lines.append(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            summary_lines.append("")
            
            # Layer-by-layer breakdown
            summary_lines.append("Layer Breakdown:")
            summary_lines.append("-" * 30)
            
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    param_count = sum(p.numel() for p in module.parameters())
                    if param_count > 0:
                        summary_lines.append(f"{name}: {module.__class__.__name__} ({param_count:,} params)")
            
            # Save as text file
            summary_text = "\n".join(summary_lines)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(summary_text)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, f"model_summaries/{model_type}_summary.txt")
            os.unlink(temp_path)
            
        except Exception as e:
            warnings.warn(f"Could not save model summary: {e}")
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        model_type: str = "gan",
        log_gpu_memory: bool = True
    ):
        """
        Log training metrics with PyTorch-specific information.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step/epoch number
            model_type: Type of training (gan, diffusion)
            log_gpu_memory: Whether to log GPU memory usage
        """
        if not self.current_run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Add PyTorch-specific metrics
        enhanced_metrics = dict(metrics)
        
        # GPU memory tracking
        if log_gpu_memory and torch.cuda.is_available():
            enhanced_metrics.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            })
        
        # Log metrics to MLflow
        mlflow.log_metrics(enhanced_metrics, step=step)
        
        # Log to Weights & Biases if enabled
        if self.enable_wandb:
            wandb.log(enhanced_metrics, step=step)
        
        # Store metrics for later analysis
        for key, value in enhanced_metrics.items():
            if key not in self.run_metrics:
                self.run_metrics[key] = []
            self.run_metrics[key].append((step, value))
    
    def log_generation_samples(
        self,
        samples: torch.Tensor,
        step: int,
        sample_type: str = "generated",
        max_samples: int = 8,
        save_3d_visualization: bool = True
    ):
        """
        Log generated 3D samples as artifacts.
        
        Args:
            samples: Generated samples tensor
            step: Training step
            sample_type: Type of samples (generated, real, etc.)
            max_samples: Maximum number of samples to save
            save_3d_visualization: Whether to create 3D visualizations
        """
        if not self.current_run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        try:
            # Convert to numpy and limit number of samples
            samples_np = samples.detach().cpu().numpy()[:max_samples]
            
            # Save raw samples
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                np.save(f.name, samples_np)
                mlflow.log_artifact(f.name, f"samples/step_{step}_{sample_type}_samples.npy")
                os.unlink(f.name)
            
            # Create visualizations
            if save_3d_visualization:
                self._create_sample_visualizations(samples_np, step, sample_type)
                
        except Exception as e:
            warnings.warn(f"Could not log generation samples: {e}")
    
    def _create_sample_visualizations(
        self,
        samples: np.ndarray,
        step: int,
        sample_type: str
    ):
        """Create and save visualizations of 3D samples."""
        try:
            # Create matplotlib visualizations
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for i, sample in enumerate(samples[:8]):
                if i >= len(axes):
                    break
                
                # Take a slice through the middle of the 3D volume
                if len(sample.shape) == 4:  # (D, H, W, C)
                    slice_data = sample[sample.shape[0]//2, :, :, 0]
                elif len(sample.shape) == 3:  # (D, H, W)
                    slice_data = sample[sample.shape[0]//2, :, :]
                else:
                    continue
                
                axes[i].imshow(slice_data, cmap='viridis')
                axes[i].set_title(f'Sample {i+1}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'{sample_type.title()} Samples - Step {step}')
            
            # Save matplotlib figure
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, f"visualizations/step_{step}_{sample_type}_samples.png")
                os.unlink(f.name)
            
            plt.close(fig)
            
            # Create interactive 3D visualization if Plotly is available
            if PLOTLY_AVAILABLE and len(samples) > 0:
                self._create_3d_plotly_visualization(samples[0], step, sample_type)
                
        except Exception as e:
            warnings.warn(f"Could not create sample visualizations: {e}")
    
    def _create_3d_plotly_visualization(
        self,
        sample: np.ndarray,
        step: int,
        sample_type: str
    ):
        """Create interactive 3D visualization using Plotly."""
        try:
            # Extract non-zero voxels for 3D scatter plot
            if len(sample.shape) == 4:  # (D, H, W, C)
                volume = sample[:, :, :, 0]
            else:  # (D, H, W)
                volume = sample
            
            # Find non-zero voxels
            coords = np.where(volume > 0.1)  # Threshold for visualization
            if len(coords[0]) == 0:
                return
            
            values = volume[coords]
            
            # Create 3D scatter plot
            fig = go.Figure(data=go.Scatter3d(
                x=coords[0],
                y=coords[1],
                z=coords[2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=values,
                    colorscale='Viridis',
                    opacity=0.8
                )
            ))
            
            fig.update_layout(
                title=f'{sample_type.title()} 3D Sample - Step {step}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )
            
            # Save as HTML
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                fig.write_html(f.name)
                mlflow.log_artifact(f.name, f"visualizations/step_{step}_{sample_type}_3d.html")
                os.unlink(f.name)
                
        except Exception as e:
            warnings.warn(f"Could not create 3D Plotly visualization: {e}")
    
    def log_model_comparison(
        self,
        pytorch_metrics: Dict[str, float],
        tensorflow_metrics: Optional[Dict[str, float]] = None,
        comparison_name: str = "framework_comparison"
    ):
        """
        Log comparison between PyTorch and TensorFlow models.
        
        Args:
            pytorch_metrics: Metrics from PyTorch model
            tensorflow_metrics: Metrics from TensorFlow model (optional)
            comparison_name: Name for the comparison
        """
        if not self.current_run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        comparison_data = {
            "pytorch_metrics": pytorch_metrics,
            "tensorflow_metrics": tensorflow_metrics or {},
            "comparison_timestamp": datetime.now().isoformat(),
        }
        
        # Calculate improvement metrics if both are available
        if tensorflow_metrics:
            improvements = {}
            for key in pytorch_metrics:
                if key in tensorflow_metrics:
                    tf_val = tensorflow_metrics[key]
                    pt_val = pytorch_metrics[key]
                    if tf_val != 0:
                        improvement = (tf_val - pt_val) / tf_val * 100
                        improvements[f"{key}_improvement_percent"] = improvement
            
            comparison_data["improvements"] = improvements
            mlflow.log_metrics(improvements)
        
        # Save comparison as JSON artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(comparison_data, f, indent=2)
            mlflow.log_artifact(f.name, f"comparisons/{comparison_name}.json")
            os.unlink(f.name)
        
        # Create comparison visualization
        if tensorflow_metrics:
            self._create_comparison_plot(pytorch_metrics, tensorflow_metrics, comparison_name)
    
    def _create_comparison_plot(
        self,
        pytorch_metrics: Dict[str, float],
        tensorflow_metrics: Dict[str, float],
        comparison_name: str
    ):
        """Create comparison plot between frameworks."""
        try:
            # Find common metrics
            common_keys = set(pytorch_metrics.keys()) & set(tensorflow_metrics.keys())
            if not common_keys:
                return
            
            # Create comparison bar plot
            metrics = list(common_keys)
            pytorch_values = [pytorch_metrics[k] for k in metrics]
            tensorflow_values = [tensorflow_metrics[k] for k in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, pytorch_values, width, label='PyTorch', alpha=0.8)
            ax.bar(x + width/2, tensorflow_values, width, label='TensorFlow', alpha=0.8)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title('PyTorch vs TensorFlow Model Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, f"comparisons/{comparison_name}_plot.png")
                os.unlink(f.name)
            
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"Could not create comparison plot: {e}")
    
    def log_sparse_tensor_metrics(
        self,
        tensor: torch.Tensor,
        tensor_name: str = "tensor",
        step: Optional[int] = None
    ):
        """
        Log metrics specific to sparse tensors.
        
        Args:
            tensor: Tensor to analyze
            tensor_name: Name of the tensor
            step: Optional step number
        """
        if not self.current_run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        try:
            # Calculate sparsity metrics
            if tensor.is_sparse:
                total_elements = tensor.numel()
                nnz = tensor._nnz()
                sparsity = 1.0 - (nnz / total_elements)
                
                sparse_metrics = {
                    f"{tensor_name}_sparsity": sparsity,
                    f"{tensor_name}_nnz": nnz,
                    f"{tensor_name}_total_elements": total_elements,
                    f"{tensor_name}_memory_saved_ratio": sparsity,
                }
            else:
                # Dense tensor - calculate potential sparsity
                zero_elements = (tensor == 0).sum().item()
                total_elements = tensor.numel()
                potential_sparsity = zero_elements / total_elements
                
                sparse_metrics = {
                    f"{tensor_name}_potential_sparsity": potential_sparsity,
                    f"{tensor_name}_zero_elements": zero_elements,
                    f"{tensor_name}_total_elements": total_elements,
                    f"{tensor_name}_is_sparse": False,
                }
            
            # Log metrics
            if step is not None:
                mlflow.log_metrics(sparse_metrics, step=step)
            else:
                mlflow.log_metrics(sparse_metrics)
            
            if self.enable_wandb:
                wandb.log(sparse_metrics, step=step)
                
        except Exception as e:
            warnings.warn(f"Could not log sparse tensor metrics: {e}")
    
    def save_model_to_registry(
        self,
        model: nn.Module,
        model_type: str = "generator",
        stage: str = "Staging",
        description: Optional[str] = None
    ):
        """
        Save PyTorch model to MLflow model registry.
        
        Args:
            model: PyTorch model to save
            model_type: Type of model
            stage: Model stage (Staging, Production)
            description: Optional model description
        """
        if not self.current_run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        try:
            model_name = f"{self.model_name}_{model_type}"
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"models/{model_type}",
                registered_model_name=model_name,
            )
            
            # Save model state dict as additional artifact
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                mlflow.log_artifact(f.name, f"model_state_dicts/{model_type}_state_dict.pth")
                os.unlink(f.name)
            
            # Update model stage if specified
            if stage != "None":
                client = MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=["None"])
                if latest_version:
                    version = latest_version[0].version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version,
                        stage=stage
                    )
                    
                    # Add description if provided
                    if description:
                        client.update_model_version(
                            name=model_name,
                            version=version,
                            description=description
                        )
            
            print(f"✅ Model {model_name} saved to registry in {stage} stage")
            
        except Exception as e:
            warnings.warn(f"Could not save model to registry: {e}")
    
    def create_training_summary(self) -> Dict[str, Any]:
        """
        Create a comprehensive training summary.
        
        Returns:
            Dictionary containing training summary
        """
        if not self.run_metrics:
            return {}
        
        summary = {
            "run_id": self.current_run.info.run_id if self.current_run else None,
            "experiment_name": self.experiment_name,
            "total_metrics_logged": len(self.run_metrics),
            "metrics_summary": {},
            "model_artifacts": list(self.model_artifacts.keys()),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Calculate summary statistics for each metric
        for metric_name, values in self.run_metrics.items():
            if values:
                metric_values = [v[1] for v in values]  # Extract values (ignore steps)
                summary["metrics_summary"][metric_name] = {
                    "final_value": metric_values[-1],
                    "best_value": min(metric_values) if "loss" in metric_name.lower() else max(metric_values),
                    "mean_value": np.mean(metric_values),
                    "std_value": np.std(metric_values),
                    "total_steps": len(values),
                }
        
        # Save summary as artifact
        if self.current_run:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(summary, f, indent=2)
                mlflow.log_artifact(f.name, "training_summary.json")
                os.unlink(f.name)
        
        return summary


def create_pytorch_mlflow_tracker(
    experiment_name: str = "deepSculpt_pytorch",
    **kwargs
) -> PyTorchMLflowTracker:
    """
    Factory function to create a PyTorch MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for the tracker
        
    Returns:
        Configured PyTorchMLflowTracker instance
    """
    return PyTorchMLflowTracker(experiment_name=experiment_name, **kwargs)


# Utility functions for backward compatibility
def log_pytorch_model(
    model: nn.Module,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    model_type: str = "generator",
    experiment_name: str = "deepSculpt_pytorch"
):
    """
    Simplified function to log a PyTorch model with metrics and parameters.
    
    Args:
        model: PyTorch model to log
        metrics: Optional metrics dictionary
        params: Optional parameters dictionary
        model_type: Type of model
        experiment_name: MLflow experiment name
    """
    tracker = create_pytorch_mlflow_tracker(experiment_name)
    
    with tracker:
        run_id = tracker.start_run()
        
        # Log model architecture
        tracker.log_model_architecture(model, model_type)
        
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)
        
        # Save model to registry
        tracker.save_model_to_registry(model, model_type)
        
        tracker.end_run()
    
    return run_id


# Context manager support
class PyTorchMLflowTracker:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()