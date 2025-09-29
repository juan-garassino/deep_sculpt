#!/usr/bin/env python3
"""
Comprehensive Monitoring and Logging for DeepSculpt v2.0

Advanced monitoring system with:
- GPU utilization tracking
- Real-time performance monitoring
- Memory usage optimization
- Training progress visualization
- System health checks
- Automated performance regression detection
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from collections import deque, defaultdict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    
    # GPU metrics (if available)
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    
    # Process-specific metrics
    process_cpu_percent: Optional[float] = None
    process_memory_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_available_gb': self.memory_available_gb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'gpu_utilization': self.gpu_utilization,
            'gpu_memory_used_gb': self.gpu_memory_used_gb,
            'gpu_memory_total_gb': self.gpu_memory_total_gb,
            'gpu_temperature': self.gpu_temperature,
            'process_cpu_percent': self.process_cpu_percent,
            'process_memory_gb': self.process_memory_gb
        }


@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    epoch: int
    step: int
    timestamp: float
    
    # Loss metrics
    loss: Optional[float] = None
    gen_loss: Optional[float] = None
    disc_loss: Optional[float] = None
    
    # Performance metrics
    samples_per_second: Optional[float] = None
    time_per_epoch: Optional[float] = None
    time_per_step: Optional[float] = None
    
    # Memory metrics
    peak_memory_gb: Optional[float] = None
    current_memory_gb: Optional[float] = None
    
    # Learning metrics
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class GPUMonitor:
    """GPU utilization and memory monitoring."""
    
    def __init__(self):
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_gpu_metrics(self, device_id: int = 0) -> Dict[str, Optional[float]]:
        """Get GPU metrics for specified device."""
        if not self.gpu_available or device_id >= self.device_count:
            return {
                'utilization': None,
                'memory_used_gb': None,
                'memory_total_gb': None,
                'temperature': None
            }
        
        try:
            # Memory metrics
            memory_used = torch.cuda.memory_allocated(device_id) / 1e9
            memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            
            # Try to get utilization (may not be available on all systems)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                utilization = None
                temperature = None
            
            return {
                'utilization': utilization,
                'memory_used_gb': memory_used,
                'memory_total_gb': memory_total,
                'temperature': temperature
            }
            
        except Exception as e:
            warnings.warn(f"Failed to get GPU metrics: {e}")
            return {
                'utilization': None,
                'memory_used_gb': None,
                'memory_total_gb': None,
                'temperature': None
            }
    
    def get_all_gpu_metrics(self) -> List[Dict[str, Optional[float]]]:
        """Get metrics for all available GPUs."""
        return [self.get_gpu_metrics(i) for i in range(self.device_count)]


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.gpu_monitor = GPUMonitor()
        self.process = psutil.Process()
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        # GPU metrics
        gpu_metrics = self.gpu_monitor.get_gpu_metrics(0)
        
        # Process metrics
        try:
            process_cpu = self.process.cpu_percent()
            process_memory = self.process.memory_info().rss / 1e9
        except:
            process_cpu = None
            process_memory = None
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1e9,
            memory_available_gb=memory.available / 1e9,
            disk_usage_percent=(disk.used / disk.total) * 100,
            disk_free_gb=disk.free / 1e9,
            gpu_utilization=gpu_metrics['utilization'],
            gpu_memory_used_gb=gpu_metrics['memory_used_gb'],
            gpu_memory_total_gb=gpu_metrics['memory_total_gb'],
            gpu_temperature=gpu_metrics['temperature'],
            process_cpu_percent=process_cpu,
            process_memory_gb=process_memory
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """Get summary statistics for recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-window_size:]
        
        # Extract numeric values
        cpu_values = [m.cpu_percent for m in recent_metrics if m.cpu_percent is not None]
        memory_values = [m.memory_percent for m in recent_metrics if m.memory_percent is not None]
        gpu_util_values = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization is not None]
        gpu_memory_values = [m.gpu_memory_used_gb for m in recent_metrics if m.gpu_memory_used_gb is not None]
        
        summary = {
            'window_size': len(recent_metrics),
            'time_span_minutes': (recent_metrics[-1].timestamp - recent_metrics[0].timestamp) / 60 if len(recent_metrics) > 1 else 0
        }
        
        # CPU statistics
        if cpu_values:
            summary['cpu'] = {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            }
        
        # Memory statistics
        if memory_values:
            summary['memory'] = {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            }
        
        # GPU statistics
        if gpu_util_values:
            summary['gpu_utilization'] = {
                'mean': np.mean(gpu_util_values),
                'max': np.max(gpu_util_values),
                'min': np.min(gpu_util_values),
                'std': np.std(gpu_util_values)
            }
        
        if gpu_memory_values:
            summary['gpu_memory'] = {
                'mean': np.mean(gpu_memory_values),
                'max': np.max(gpu_memory_values),
                'min': np.min(gpu_memory_values),
                'std': np.std(gpu_memory_values)
            }
        
        return summary


class TrainingMonitor:
    """Training progress monitoring and visualization."""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.training_history: deque = deque(maxlen=history_size)
        self.system_monitor = SystemMonitor()
        
        # Performance tracking
        self.epoch_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None
        self.samples_processed = 0
        
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        
    def start_step(self, step: int):
        """Mark the start of a training step."""
        self.step_start_time = time.time()
        self.current_step = step
        
    def log_training_step(self, metrics: Dict[str, Any], batch_size: int = 1):
        """Log training step metrics."""
        current_time = time.time()
        
        # Calculate performance metrics
        time_per_step = current_time - self.step_start_time if self.step_start_time else None
        samples_per_second = batch_size / time_per_step if time_per_step and time_per_step > 0 else None
        
        # Get system metrics
        system_metrics = self.system_monitor.get_system_metrics()
        
        # Create training metrics
        training_metrics = TrainingMetrics(
            epoch=getattr(self, 'current_epoch', 0),
            step=getattr(self, 'current_step', 0),
            timestamp=current_time,
            loss=metrics.get('loss'),
            gen_loss=metrics.get('gen_loss'),
            disc_loss=metrics.get('disc_loss'),
            samples_per_second=samples_per_second,
            time_per_step=time_per_step,
            peak_memory_gb=system_metrics.gpu_memory_used_gb,
            current_memory_gb=system_metrics.gpu_memory_used_gb,
            learning_rate=metrics.get('learning_rate'),
            gradient_norm=metrics.get('gradient_norm')
        )
        
        self.training_history.append(training_metrics)
        self.samples_processed += batch_size
        
        return training_metrics
    
    def end_epoch(self):
        """Mark the end of an epoch."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            # Update last training metric with epoch time
            if self.training_history:
                last_metric = self.training_history[-1]
                last_metric.time_per_epoch = epoch_time
    
    def get_training_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """Get training performance summary."""
        if not self.training_history:
            return {}
        
        recent_metrics = list(self.training_history)[-window_size:]
        
        # Extract values
        losses = [m.loss for m in recent_metrics if m.loss is not None]
        gen_losses = [m.gen_loss for m in recent_metrics if m.gen_loss is not None]
        disc_losses = [m.disc_loss for m in recent_metrics if m.disc_loss is not None]
        throughput = [m.samples_per_second for m in recent_metrics if m.samples_per_second is not None]
        
        summary = {
            'total_samples_processed': self.samples_processed,
            'window_size': len(recent_metrics)
        }
        
        if losses:
            summary['loss'] = {
                'current': losses[-1],
                'mean': np.mean(losses),
                'min': np.min(losses),
                'trend': 'decreasing' if len(losses) > 10 and losses[-1] < np.mean(losses[:10]) else 'stable'
            }
        
        if gen_losses:
            summary['gen_loss'] = {
                'current': gen_losses[-1],
                'mean': np.mean(gen_losses),
                'min': np.min(gen_losses)
            }
        
        if disc_losses:
            summary['disc_loss'] = {
                'current': disc_losses[-1],
                'mean': np.mean(disc_losses),
                'min': np.min(disc_losses)
            }
        
        if throughput:
            summary['throughput'] = {
                'current_samples_per_sec': throughput[-1],
                'mean_samples_per_sec': np.mean(throughput),
                'max_samples_per_sec': np.max(throughput)
            }
        
        return summary


class PerformanceRegressor:
    """Automated performance regression detection."""
    
    def __init__(self, baseline_window: int = 100, comparison_window: int = 50):
        self.baseline_window = baseline_window
        self.comparison_window = comparison_window
        self.baselines: Dict[str, List[float]] = defaultdict(list)
        
    def update_baseline(self, metric_name: str, value: float):
        """Update baseline values for a metric."""
        self.baselines[metric_name].append(value)
        if len(self.baselines[metric_name]) > self.baseline_window:
            self.baselines[metric_name].pop(0)
    
    def check_regression(self, metric_name: str, recent_values: List[float], 
                        threshold: float = 0.1) -> Dict[str, Any]:
        """Check for performance regression."""
        if metric_name not in self.baselines or len(self.baselines[metric_name]) < 10:
            return {'regression_detected': False, 'reason': 'insufficient_baseline_data'}
        
        if len(recent_values) < 5:
            return {'regression_detected': False, 'reason': 'insufficient_recent_data'}
        
        baseline_mean = np.mean(self.baselines[metric_name])
        recent_mean = np.mean(recent_values)
        
        # For metrics where lower is better (like loss)
        if metric_name.endswith('_loss') or metric_name == 'loss':
            regression = recent_mean > baseline_mean * (1 + threshold)
            improvement = recent_mean < baseline_mean * (1 - threshold)
        else:
            # For metrics where higher is better (like throughput)
            regression = recent_mean < baseline_mean * (1 - threshold)
            improvement = recent_mean > baseline_mean * (1 + threshold)
        
        change_percent = ((recent_mean - baseline_mean) / baseline_mean) * 100
        
        return {
            'regression_detected': regression,
            'improvement_detected': improvement,
            'baseline_mean': baseline_mean,
            'recent_mean': recent_mean,
            'change_percent': change_percent,
            'threshold_percent': threshold * 100
        }


class RealTimeVisualizer:
    """Real-time monitoring visualization."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, save_path: Optional[str] = None):
        """Start real-time monitoring."""
        if self.running:
            return
        
        self.running = True
        self.save_path = save_path
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                system_metrics = self.system_monitor.get_system_metrics()
                
                # Save metrics if path provided
                if self.save_path:
                    self._save_metrics(system_metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _save_metrics(self, metrics: SystemMetrics):
        """Save metrics to file."""
        try:
            metrics_file = Path(self.save_path) / "monitoring_metrics.jsonl"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
                
        except Exception as e:
            warnings.warn(f"Failed to save metrics: {e}")
    
    def create_dashboard(self, output_path: str = "monitoring_dashboard.html"):
        """Create interactive monitoring dashboard."""
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available - cannot create dashboard")
            return
        
        # Get recent metrics
        recent_metrics = list(self.system_monitor.metrics_history)[-100:]
        if not recent_metrics:
            warnings.warn("No metrics available for dashboard")
            return
        
        # Extract data
        timestamps = [m.timestamp for m in recent_metrics]
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_util = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization is not None]
        gpu_memory = [m.gpu_memory_used_gb for m in recent_metrics if m.gpu_memory_used_gb is not None]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'GPU Utilization', 'GPU Memory'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU plot
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_values, name='CPU %', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Memory plot
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_values, name='Memory %', line=dict(color='green')),
            row=1, col=2
        )
        
        # GPU plots (if available)
        if gpu_util:
            fig.add_trace(
                go.Scatter(x=timestamps[:len(gpu_util)], y=gpu_util, name='GPU %', line=dict(color='red')),
                row=2, col=1
            )
        
        if gpu_memory:
            fig.add_trace(
                go.Scatter(x=timestamps[:len(gpu_memory)], y=gpu_memory, name='GPU Memory GB', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="DeepSculpt v2.0 - System Monitoring Dashboard",
            showlegend=True,
            height=600
        )
        
        # Save dashboard
        fig.write_html(output_path)
        print(f"📊 Dashboard saved to: {output_path}")


class ExperimentTracker:
    """Experiment tracking integration."""
    
    def __init__(self, tracking_backend: str = "local", experiment_name: str = "deepsculpt_v2"):
        self.tracking_backend = tracking_backend
        self.experiment_name = experiment_name
        self.active_run = None
        
        # Initialize tracking backend
        if tracking_backend == "mlflow" and MLFLOW_AVAILABLE:
            self._init_mlflow()
        elif tracking_backend == "wandb" and WANDB_AVAILABLE:
            self._init_wandb()
        elif tracking_backend == "local":
            self._init_local()
        else:
            warnings.warn(f"Tracking backend '{tracking_backend}' not available, using local")
            self._init_local()
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            print(f"📊 MLflow experiment: {self.experiment_name}")
        except Exception as e:
            warnings.warn(f"MLflow initialization failed: {e}")
            self._init_local()
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            wandb.init(project=self.experiment_name)
            print(f"📊 W&B project: {self.experiment_name}")
        except Exception as e:
            warnings.warn(f"W&B initialization failed: {e}")
            self._init_local()
    
    def _init_local(self):
        """Initialize local tracking."""
        self.local_log_path = Path("./experiment_logs")
        self.local_log_path.mkdir(exist_ok=True)
        print(f"📊 Local tracking: {self.local_log_path}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to tracking backend."""
        if self.tracking_backend == "mlflow" and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        
        elif self.tracking_backend == "wandb" and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
        
        else:
            # Local logging
            log_entry = {
                'timestamp': time.time(),
                'step': step,
                'metrics': metrics
            }
            
            log_file = self.local_log_path / "metrics.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def log_system_metrics(self, system_metrics: SystemMetrics, step: Optional[int] = None):
        """Log system metrics."""
        metrics_dict = {
            'system/cpu_percent': system_metrics.cpu_percent,
            'system/memory_percent': system_metrics.memory_percent,
            'system/memory_used_gb': system_metrics.memory_used_gb,
            'system/disk_usage_percent': system_metrics.disk_usage_percent
        }
        
        if system_metrics.gpu_utilization is not None:
            metrics_dict['system/gpu_utilization'] = system_metrics.gpu_utilization
        
        if system_metrics.gpu_memory_used_gb is not None:
            metrics_dict['system/gpu_memory_used_gb'] = system_metrics.gpu_memory_used_gb
        
        self.log_metrics(metrics_dict, step=step)
    
    def finish(self):
        """Finish experiment tracking."""
        if self.tracking_backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.end_run()
        elif self.tracking_backend == "wandb" and WANDB_AVAILABLE:
            wandb.finish()


# Main monitoring interface
class DeepSculptMonitor:
    """Main monitoring interface for DeepSculpt v2.0."""
    
    def __init__(self, 
                 tracking_backend: str = "local",
                 experiment_name: str = "deepsculpt_v2",
                 enable_real_time: bool = True,
                 save_path: Optional[str] = "./monitoring"):
        
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor()
        self.performance_regressor = PerformanceRegressor()
        self.experiment_tracker = ExperimentTracker(tracking_backend, experiment_name)
        
        if enable_real_time:
            self.visualizer = RealTimeVisualizer()
            if save_path:
                self.visualizer.start_monitoring(save_path)
        else:
            self.visualizer = None
    
    def start_training_monitoring(self, epoch: int):
        """Start monitoring for training epoch."""
        self.training_monitor.start_epoch(epoch)
    
    def log_training_step(self, metrics: Dict[str, Any], step: int, batch_size: int = 1):
        """Log training step with comprehensive monitoring."""
        self.training_monitor.start_step(step)
        
        # Log training metrics
        training_metrics = self.training_monitor.log_training_step(metrics, batch_size)
        
        # Get system metrics
        system_metrics = self.system_monitor.get_system_metrics()
        
        # Log to experiment tracker
        combined_metrics = {
            **metrics,
            'performance/samples_per_second': training_metrics.samples_per_second,
            'performance/time_per_step': training_metrics.time_per_step
        }
        
        self.experiment_tracker.log_metrics(combined_metrics, step=step)
        self.experiment_tracker.log_system_metrics(system_metrics, step=step)
        
        # Check for performance regressions
        if training_metrics.samples_per_second:
            self.performance_regressor.update_baseline('throughput', training_metrics.samples_per_second)
        
        return training_metrics
    
    def end_training_monitoring(self):
        """End training monitoring."""
        self.training_monitor.end_epoch()
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            'system': self.system_monitor.get_metrics_summary(),
            'training': self.training_monitor.get_training_summary(),
            'timestamp': time.time()
        }
    
    def create_monitoring_report(self, output_path: str = "./monitoring_report.html"):
        """Create comprehensive monitoring report."""
        if self.visualizer:
            self.visualizer.create_dashboard(output_path)
        else:
            print("Real-time visualizer not enabled - cannot create dashboard")
    
    def shutdown(self):
        """Shutdown monitoring systems."""
        if self.visualizer:
            self.visualizer.stop_monitoring()
        self.experiment_tracker.finish()


# Convenience functions
def create_monitor(tracking_backend: str = "local", 
                  experiment_name: str = "deepsculpt_v2",
                  enable_real_time: bool = True) -> DeepSculptMonitor:
    """Create a DeepSculpt monitor with default settings."""
    return DeepSculptMonitor(
        tracking_backend=tracking_backend,
        experiment_name=experiment_name,
        enable_real_time=enable_real_time
    )


def quick_system_check() -> Dict[str, Any]:
    """Quick system health check."""
    monitor = SystemMonitor()
    metrics = monitor.get_system_metrics()
    
    # Health assessment
    health = {
        'overall': 'good',
        'warnings': [],
        'recommendations': []
    }
    
    # Check CPU
    if metrics.cpu_percent > 90:
        health['warnings'].append('High CPU usage')
        health['overall'] = 'warning'
    
    # Check memory
    if metrics.memory_percent > 85:
        health['warnings'].append('High memory usage')
        health['recommendations'].append('Consider reducing batch size')
        health['overall'] = 'warning'
    
    # Check disk
    if metrics.disk_usage_percent > 90:
        health['warnings'].append('Low disk space')
        health['recommendations'].append('Clean up temporary files')
        health['overall'] = 'warning'
    
    # Check GPU
    if metrics.gpu_memory_used_gb and metrics.gpu_memory_total_gb:
        gpu_usage_percent = (metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb) * 100
        if gpu_usage_percent > 90:
            health['warnings'].append('High GPU memory usage')
            health['recommendations'].append('Enable sparse tensors or reduce model size')
            health['overall'] = 'warning'
    
    return {
        'metrics': metrics.to_dict(),
        'health': health,
        'timestamp': time.time()
    }