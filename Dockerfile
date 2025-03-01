FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN useradd -m -u 1000 user

RUN mkdir -p /app && chown user /app

WORKDIR /app

COPY --chown=user . /app

USER user

ENV HOME=/home/user

RUN uv sync

CMD [".venv/bin/fastapi", "run", "--port", "7860", "main.py"]
