FROM node:lts-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/ ./
RUN npm ci
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN useradd -m -u 1000 user

RUN mkdir -p /app && chown user /app

WORKDIR /app

COPY --chown=user . /app
COPY --from=frontend-builder --chown=user /frontend/public /app/frontend/public

USER user

ENV HOME=/home/user

RUN uv sync

CMD ["uv", "run", "fastapi", "run", "--port", "7860", "main.py"]
