import os
from datetime import datetime
from typing import Dict, Any

import mlflow


def get_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required")
    return uri


def get_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "deepsculpt")


def start_run(params: Dict[str, Any]) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(get_experiment_name())

    run_name = params.get("run_name") or f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run = mlflow.start_run(run_name=run_name)
    for key, value in params.items():
        if value is None:
            continue
        mlflow.log_param(key, value)
    return run


def log_metric(metric_name: str, value: float, step: int | None = None) -> None:
    if step is None:
        mlflow.log_metric(metric_name, value)
    else:
        mlflow.log_metric(metric_name, value, step=step)


def end_run() -> None:
    mlflow.end_run()
