FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY knn ./knn
RUN python -m knn.main


FROM python:3.13-slim

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 1000 api

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/knn/predict ./knn/predict
COPY --from=builder /app/knn/utils ./knn/utils
COPY --from=builder /app/knn/model ./knn/model
COPY app ./app

USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
