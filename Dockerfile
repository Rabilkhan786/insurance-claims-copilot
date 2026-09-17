FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "[ -f data/crm.db ] || python Data/seed.py; exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
