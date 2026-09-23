FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

# `--no-dev` keeps `uv run`'s implicit environment sync (which otherwise
# reconciles against every dependency group on every start) from pulling
# in lint/type-check tooling inside the runtime image.
CMD ["uv", "run", "--no-dev", "uvicorn", "apps.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
