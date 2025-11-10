"""
DeepSculpt v2.0 Workflow Management

PyTorch-based workflow orchestration and experiment tracking.
"""

# Make workflow imports optional
try:
    from .pytorch_workflow import PyTorchManager
    WORKFLOW_AVAILABLE = True
except ImportError:
    PyTorchManager = None
    WORKFLOW_AVAILABLE = False

try:
    from .pytorch_mlflow_tracking import PyTorchMLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    PyTorchMLflowTracker = None
    MLFLOW_AVAILABLE = False

__all__ = [
    "PyTorchManager",
    "PyTorchMLflowTracker",
    "WORKFLOW_AVAILABLE",
    "MLFLOW_AVAILABLE"
]