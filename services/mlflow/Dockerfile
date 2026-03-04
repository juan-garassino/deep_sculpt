FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/mlflow/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port ${PORT:-8080} --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${ARTIFACT_ROOT}"]
