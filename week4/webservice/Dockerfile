FROM python:3.13.2-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY ["pyproject.toml", "./uv.lock", "./"]

RUN uv sync --locked
# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]