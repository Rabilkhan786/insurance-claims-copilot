# Same Python as pyproject.toml's requires-python, and the same installer as
# local development. The image used to run `pip install -r requirements.txt`
# on python:3.14-slim while development ran uv on 3.11, so the two
# environments resolved different wheels and only one of them was ever tested.
FROM python:3.11-slim

# uv reads pyproject.toml + uv.lock, which is the single dependency source.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --frozen fails the build if uv.lock and pyproject.toml disagree, rather
# than silently resolving something the lockfile never pinned.
# --no-dev leaves pytest out of the runtime image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# The demo databases are gitignored, so the image builds them at start-up
# rather than shipping an empty crm.db that every lookup would miss on.
# Seeding is idempotent: seed.py drops and rebuilds the file.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "python Data/seed.py && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
