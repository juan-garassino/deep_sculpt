import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any

from google.cloud import storage


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    without_scheme = uri[len("gs://") :]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob


def write_pointer(latest_pointer_uri: str, payload: Dict[str, Any]) -> None:
    bucket_name, blob_name = _parse_gcs_uri(latest_pointer_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name

    try:
        blob.upload_from_filename(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def read_pointer(latest_pointer_uri: str) -> Dict[str, Any]:
    bucket_name, blob_name = _parse_gcs_uri(latest_pointer_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return {}

    content = blob.download_as_text()
    return json.loads(content)
