FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . /app

RUN uv sync

ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

CMD [".venv/bin/fastapi", "run", "--port", "7860", "main.py"]
