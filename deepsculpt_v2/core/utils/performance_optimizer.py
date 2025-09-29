#!/usr/bin/env python3
"""
Performance Optimization Utilities for DeepSculpt v2.0

Automated performance optimization with:
- Dynamic batch size adjustment
- Memory optimization suggestions
- GPU utilization optimization
- Training speed improvements
- Resource allocation optimization
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .monitoring import SystemMonitor, GPUMonitor


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    category: str  # 'memory', 'speed', 'gpu', 'data'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    action: str
    expected_improvement: str
    implementation_difficulty: str  # 'easy', 'medium', 'hard'
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'action': self.action,
            'expected_improvement': self.expected_improvement,
            'implementation_difficulty': self.implementation_difficulty
        }


class BatchSizeOptimizer:
    """Dynamic batch size optimization."""
    
    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 1, max_batch_size: int = 128):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.oom_count = 0
        
    def suggest_batch_size(self, available_memory_gb: float, model_size_gb: float) -> int:
        """Suggest optimal batch size based on available memory."""
        # Rough estimation: leave 20% memory free, account for gradients (2x model size)
        usable_memory = available_memory_gb * 0.8
        memory_per_sample = model_size_gb * 2  # Model + gradients
        
        suggested_batch_size = max(1, int(usable_memory / memory_per_sample))
        
        # Clamp to valid range
        suggested_batch_size = max(self.min_batch_size, 
                                 min(self.max_batch_size, suggested_batch_size))
        
        return suggested_batch_size
    
    def handle_oom_error(self) -> int:
        """Handle out-of-memory error by reducing batch size."""
        self.oom_count += 1
        
        # Reduce batch size by 50% or to minimum
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        if new_batch_size == self.current_batch_size:
            # Already at minimum, can't reduce further
            warnings.warn("Cannot reduce batch size further - consider using smaller model or sparse tensors")
            return self.current_batch_size
        
        self.current_batch_size = new_batch_size
        print(f"⚠️ OOM detected - reducing batch size to {new_batch_size}")
        
        return new_batch_size
    
    def optimize_batch_size(self, throughput_history: List[float], memory_usage_history: List[float]) -> int:
        """Optimize batch size based on performance history."""
        if len(throughput_history) < 5 or len(memory_usage_history) < 5:
            return self.current_batch_size
        
        recent_throughput = sum(throughput_history[-5:]) / 5
        recent_memory = sum(memory_usage_history[-5:]) / 5
        
        # If memory usage is low and throughput is stable, try increasing batch size
        if recent_memory < 0.7 and self.current_batch_size < self.max_batch_size:
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            print(f"📈 Low memory usage detected - increasing batch size to {new_batch_size}")
            self.current_batch_size = new_batch_size
            
        # If memory usage is high, reduce batch size
        elif recent_memory > 0.9 and self.current_batch_size > self.min_batch_size:
            new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            print(f"📉 High memory usage detected - reducing batch size to {new_batch_size}")
            self.current_batch_size = new_batch_size
        
        return self.current_batch_size


class MemoryOptimizer:
    """Memory usage optimization."""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor() if TORCH_AVAILABLE else None
        
    def analyze_memory_usage(self, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Analyze current memory usage and provide insights."""
        analysis = {
            'timestamp': time.time(),
            'recommendations': []
        }
        
        if not TORCH_AVAILABLE:
            analysis['error'] = 'PyTorch not available'
            return analysis
        
        # GPU memory analysis
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            analysis['gpu_memory'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100,
                'fragmentation_gb': reserved - allocated
            }
            
            # Memory recommendations
            if allocated / total > 0.9:
                analysis['recommendations'].append({
                    'type': 'memory',
                    'priority': 'high',
                    'message': 'GPU memory usage very high - consider reducing batch size or using sparse tensors'
                })
            
            if (reserved - allocated) / total > 0.1:
                analysis['recommendations'].append({
                    'type': 'memory',
                    'priority': 'medium',
                    'message': 'High memory fragmentation detected - consider calling torch.cuda.empty_cache()'
                })
        
        # Model memory analysis
        if model is not None:
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9
            
            analysis['model_memory'] = {
                'parameters_gb': param_memory,
                'buffers_gb': buffer_memory,
                'total_gb': param_memory + buffer_memory
            }
        
        return analysis
    
    def optimize_memory_usage(self) -> List[str]:
        """Apply memory optimizations and return list of actions taken."""
        actions = []
        
        if not TORCH_AVAILABLE:
            return ['PyTorch not available - cannot optimize memory']
        
        if torch.cuda.is_available():
            # Clear cache
            initial_reserved = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            final_reserved = torch.cuda.memory_reserved()
            
            if initial_reserved > final_reserved:
                freed_mb = (initial_reserved - final_reserved) / 1e6
                actions.append(f'Cleared GPU cache - freed {freed_mb:.1f} MB')
            
            # Set memory fraction if usage is high
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            
            if allocated / total > 0.8:
                try:
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    actions.append('Set GPU memory fraction to 80%')
                except:
                    pass
        
        return actions
    
    def suggest_sparse_conversion(self, tensor: torch.Tensor, threshold: float = 0.1) -> Dict[str, Any]:
        """Suggest sparse tensor conversion if beneficial."""
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        sparsity = (tensor == 0).float().mean().item()
        current_memory = tensor.numel() * tensor.element_size()
        
        # Estimate sparse memory usage (rough approximation)
        non_zero_elements = int(tensor.numel() * (1 - sparsity))
        sparse_memory = non_zero_elements * (tensor.element_size() + 8)  # value + index
        
        memory_savings = (current_memory - sparse_memory) / current_memory
        
        recommendation = {
            'sparsity': sparsity,
            'current_memory_mb': current_memory / 1e6,
            'estimated_sparse_memory_mb': sparse_memory / 1e6,
            'estimated_memory_savings_percent': memory_savings * 100,
            'recommend_sparse': sparsity > threshold and memory_savings > 0.2
        }
        
        return recommendation


class GPUOptimizer:
    """GPU utilization optimization."""
    
    def __init__(self):
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
    def optimize_gpu_settings(self) -> List[str]:
        """Optimize GPU settings for better performance."""
        actions = []
        
        if not self.gpu_available:
            return ['No GPU available - using CPU']
        
        # Enable cuDNN benchmark
        if not torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark = True
            actions.append('Enabled cuDNN benchmark mode')
        
        # Disable cuDNN deterministic for better performance
        if torch.backends.cudnn.deterministic:
            torch.backends.cudnn.deterministic = False
            actions.append('Disabled cuDNN deterministic mode for better performance')
        
        # Enable TensorFloat-32 if available
        try:
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                if not torch.backends.cuda.matmul.allow_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    actions.append('Enabled TensorFloat-32 for matrix operations')
        except:
            pass
        
        return actions
    
    def analyze_gpu_utilization(self, duration_seconds: float = 10.0) -> Dict[str, Any]:
        """Analyze GPU utilization over a period."""
        if not self.gpu_available:
            return {'error': 'No GPU available'}
        
        gpu_monitor = GPUMonitor()
        utilization_samples = []
        memory_samples = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            metrics = gpu_monitor.get_gpu_metrics(0)
            if metrics['utilization'] is not None:
                utilization_samples.append(metrics['utilization'])
            if metrics['memory_used_gb'] is not None:
                memory_samples.append(metrics['memory_used_gb'])
            time.sleep(0.5)
        
        analysis = {
            'duration_seconds': duration_seconds,
            'samples_collected': len(utilization_samples)
        }
        
        if utilization_samples:
            avg_utilization = sum(utilization_samples) / len(utilization_samples)
            analysis['gpu_utilization'] = {
                'average_percent': avg_utilization,
                'max_percent': max(utilization_samples),
                'min_percent': min(utilization_samples)
            }
            
            if avg_utilization < 50:
                analysis['recommendation'] = 'Low GPU utilization - consider increasing batch size or model complexity'
            elif avg_utilization > 95:
                analysis['recommendation'] = 'Very high GPU utilization - performance is likely optimal'
        
        if memory_samples:
            analysis['gpu_memory'] = {
                'average_gb': sum(memory_samples) / len(memory_samples),
                'max_gb': max(memory_samples),
                'min_gb': min(memory_samples)
            }
        
        return analysis


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.batch_optimizer = BatchSizeOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.gpu_optimizer = GPUOptimizer()
        self.system_monitor = SystemMonitor() if PSUTIL_AVAILABLE else None
        
    def analyze_performance(self, model: Optional[nn.Module] = None, 
                          duration_seconds: float = 10.0) -> Dict[str, Any]:
        """Comprehensive performance analysis."""
        analysis = {
            'timestamp': time.time(),
            'analysis_duration_seconds': duration_seconds
        }
        
        # Memory analysis
        memory_analysis = self.memory_optimizer.analyze_memory_usage(model)
        analysis['memory'] = memory_analysis
        
        # GPU analysis
        gpu_analysis = self.gpu_optimizer.analyze_gpu_utilization(duration_seconds)
        analysis['gpu'] = gpu_analysis
        
        # System analysis
        if self.system_monitor:
            system_metrics = self.system_monitor.get_system_metrics()
            analysis['system'] = {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'disk_usage_percent': system_metrics.disk_usage_percent
            }
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Memory recommendations
        if 'memory' in analysis and 'gpu_memory' in analysis['memory']:
            gpu_mem = analysis['memory']['gpu_memory']
            utilization = gpu_mem['utilization_percent']
            
            if utilization > 90:
                recommendations.append(OptimizationRecommendation(
                    category='memory',
                    priority='high',
                    title='High GPU Memory Usage',
                    description=f'GPU memory usage is {utilization:.1f}%, which may cause OOM errors',
                    action='Reduce batch size, enable sparse tensors, or use gradient checkpointing',
                    expected_improvement='Prevent OOM errors, enable larger models',
                    implementation_difficulty='easy'
                ))
            
            if gpu_mem['fragmentation_gb'] > 1.0:
                recommendations.append(OptimizationRecommendation(
                    category='memory',
                    priority='medium',
                    title='GPU Memory Fragmentation',
                    description=f'{gpu_mem["fragmentation_gb"]:.1f} GB of fragmented memory detected',
                    action='Call torch.cuda.empty_cache() periodically during training',
                    expected_improvement='Better memory utilization, reduced OOM risk',
                    implementation_difficulty='easy'
                ))
        
        # GPU utilization recommendations
        if 'gpu' in analysis and 'gpu_utilization' in analysis['gpu']:
            gpu_util = analysis['gpu']['gpu_utilization']
            avg_util = gpu_util['average_percent']
            
            if avg_util < 50:
                recommendations.append(OptimizationRecommendation(
                    category='speed',
                    priority='medium',
                    title='Low GPU Utilization',
                    description=f'Average GPU utilization is {avg_util:.1f}%, indicating underutilization',
                    action='Increase batch size, use mixed precision training, or increase model complexity',
                    expected_improvement='2-3x training speed improvement',
                    implementation_difficulty='easy'
                ))
            
            if avg_util > 95:
                recommendations.append(OptimizationRecommendation(
                    category='gpu',
                    priority='low',
                    title='Optimal GPU Utilization',
                    description=f'GPU utilization is {avg_util:.1f}% - performance is likely optimal',
                    action='No action needed - current settings are well optimized',
                    expected_improvement='Minimal',
                    implementation_difficulty='easy'
                ))
        
        # System recommendations
        if 'system' in analysis:
            system = analysis['system']
            
            if system['cpu_percent'] > 80:
                recommendations.append(OptimizationRecommendation(
                    category='speed',
                    priority='medium',
                    title='High CPU Usage',
                    description=f'CPU usage is {system["cpu_percent"]:.1f}%, which may bottleneck data loading',
                    action='Reduce data loader workers, enable pin_memory, or use faster storage',
                    expected_improvement='Reduced training bottlenecks',
                    implementation_difficulty='easy'
                ))
            
            if system['memory_percent'] > 85:
                recommendations.append(OptimizationRecommendation(
                    category='memory',
                    priority='high',
                    title='High System Memory Usage',
                    description=f'System memory usage is {system["memory_percent"]:.1f}%',
                    action='Reduce batch size, limit data loader workers, or close other applications',
                    expected_improvement='Prevent system slowdown and swapping',
                    implementation_difficulty='easy'
                ))
        
        return recommendations
    
    def apply_automatic_optimizations(self, model: Optional[nn.Module] = None) -> List[str]:
        """Apply safe automatic optimizations."""
        actions = []
        
        # GPU optimizations
        gpu_actions = self.gpu_optimizer.optimize_gpu_settings()
        actions.extend(gpu_actions)
        
        # Memory optimizations
        memory_actions = self.memory_optimizer.optimize_memory_usage()
        actions.extend(memory_actions)
        
        return actions
    
    def create_optimization_report(self, output_path: str = "./optimization_report.json"):
        """Create comprehensive optimization report."""
        print("🔍 Analyzing performance...")
        
        # Run analysis
        analysis = self.analyze_performance(duration_seconds=15.0)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(analysis)
        
        # Apply automatic optimizations
        auto_actions = self.apply_automatic_optimizations()
        
        # Create report
        report = {
            'timestamp': time.time(),
            'analysis': analysis,
            'recommendations': [r.to_dict() for r in recommendations],
            'automatic_optimizations_applied': auto_actions,
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority_recommendations': len([r for r in recommendations if r.priority == 'high']),
                'automatic_optimizations': len(auto_actions)
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Optimization report saved to: {output_path}")
        
        # Print summary
        print("\n🎯 Optimization Summary:")
        print(f"  Total recommendations: {report['summary']['total_recommendations']}")
        print(f"  High priority: {report['summary']['high_priority_recommendations']}")
        print(f"  Auto-optimizations applied: {report['summary']['automatic_optimizations']}")
        
        if recommendations:
            print("\n🔧 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec.title} ({rec.priority} priority)")
                print(f"     {rec.action}")
        
        return report


# Convenience functions
def quick_optimize() -> List[str]:
    """Quick automatic optimization."""
    optimizer = PerformanceOptimizer()
    return optimizer.apply_automatic_optimizations()


def analyze_and_optimize(model: Optional[nn.Module] = None, 
                        output_path: str = "./optimization_report.json") -> Dict[str, Any]:
    """Analyze performance and create optimization report."""
    optimizer = PerformanceOptimizer()
    return optimizer.create_optimization_report(output_path)


def suggest_batch_size(available_memory_gb: float, model_size_gb: float) -> int:
    """Suggest optimal batch size based on available memory."""
    optimizer = BatchSizeOptimizer()
    return optimizer.suggest_batch_size(available_memory_gb, model_size_gb)