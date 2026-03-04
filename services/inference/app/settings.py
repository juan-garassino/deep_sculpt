import os


class Settings:
    def __init__(self) -> None:
        self.project_id = os.environ.get("PROJECT_ID")
        self.region = os.environ.get("REGION")
        self.train_job_name = os.environ.get("TRAIN_JOB_NAME")
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        self.artifact_root = os.environ.get("ARTIFACT_ROOT")
        self.models_bucket = os.environ.get("MODELS_BUCKET")
        self.latest_model_pointer_path = os.environ.get(
            "LATEST_MODEL_POINTER_PATH",
            f"gs://{self.models_bucket}/models/latest.json" if self.models_bucket else None,
        )
        self.allow_gcloud_fallback = os.environ.get("ALLOW_GCLOUD_FALLBACK", "false").lower() == "true"

    def validate_for_inference(self) -> None:
        missing = []
        if not self.models_bucket:
            missing.append("MODELS_BUCKET")
        if not self.latest_model_pointer_path:
            missing.append("LATEST_MODEL_POINTER_PATH")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    def validate_for_train_trigger(self) -> None:
        missing = []
        if not self.project_id:
            missing.append("PROJECT_ID")
        if not self.region:
            missing.append("REGION")
        if not self.train_job_name:
            missing.append("TRAIN_JOB_NAME")
        if missing:
            raise RuntimeError(f"Missing required env vars for training trigger: {', '.join(missing)}")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
