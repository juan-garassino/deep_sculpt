FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

COPY requirements-colab.txt /app/requirements-colab.txt
RUN pip install --no-cache-dir -r /app/requirements-colab.txt typer click

COPY . /app
RUN pip install --no-cache-dir --no-deps -e .
RUN chmod +x /app/autoresearch/entrypoint.sh /app/scripts/autoresearch_report.py

RUN useradd -m -s /bin/bash researcher && \
    chown -R researcher:researcher /app

USER researcher
WORKDIR /app

RUN git config --global user.email "autoresearch@deepsculpt.local" && \
    git config --global user.name "deepsculpt-autoresearch" && \
    git init && git add -A && git commit -m "initial"

ENV PYTHONPATH="/app:/app/deepsculpt"
ENV AUTORESEARCH_MODE=gan
ENV AUTORESEARCH_DATA_OUT=/workspace/autoresearch_data
ENV AUTORESEARCH_RUN_OUT=/workspace/autoresearch_runs

ENTRYPOINT ["/app/autoresearch/entrypoint.sh"]
