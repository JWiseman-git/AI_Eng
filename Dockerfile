FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

RUN pip install --upgrade pip \
    && pip install --no-cache-dir ".[dependencies]"

COPY ./app /app/app