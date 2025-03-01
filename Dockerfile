FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . /app

RUN uv sync

CMD [".venv/bin/fastapi", "run", "main.py"]
