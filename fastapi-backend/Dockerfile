FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dépendances d'abord (cache Docker) à partir de pyproject.toml et uv.lock
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Code de l'application
COPY . .
RUN uv sync --frozen

EXPOSE 8000 5678

# debugpy écoute sur 5678 (VSCode "Remote Attach"), uvicorn sert l'API sur 8000
CMD ["uv", "run", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
