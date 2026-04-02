# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY tinyrl ./tinyrl
COPY configs ./configs
COPY train.py quantize.py distill.py prune.py codegen.py ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

CMD ["python", "train.py", "--help"]
