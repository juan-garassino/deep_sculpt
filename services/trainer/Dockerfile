FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/trainer/requirements.txt /app/services/trainer/requirements.txt
RUN pip install --no-cache-dir -r /app/services/trainer/requirements.txt

COPY . /app

ENV PYTHONPATH=/app

CMD ["python", "-m", "services.trainer.train_entrypoint"]
