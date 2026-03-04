# Operations

## Docker Compose

1. Build images.
```bash
make docker-build
```

2. Start services.
```bash
make docker-up
```

3. Tail logs.
```bash
make docker-logs
```

4. Stop services.
```bash
make docker-down
```

## MLflow

The MLflow server is stateless and expects:
1. `BACKEND_STORE_URI` pointing to Cloud SQL Postgres in production.
2. `ARTIFACT_ROOT` pointing to GCS.

For local dev, `docker-compose.yml` uses `sqlite:///mlflow.db` and `./mlruns`.

## Cloud Run Job Execution Status

Use:
```bash
curl http://localhost:8081/train/status/projects/<PROJECT_ID>/locations/<REGION>/jobs/<JOB_NAME>/executions/<EXECUTION_ID>
```
