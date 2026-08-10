# syntax=docker/dockerfile:1

# ---- builder: install the project + its runtime dependencies into a prefix ----
FROM python:3.10-slim AS builder
WORKDIR /build

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ---- runtime: copy only what's needed to train ----
FROM python:3.10-slim AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

RUN mkdir -p models outputs/runs

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--config", "configs/train.yaml"]
