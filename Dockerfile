# syntax=docker/dockerfile:1
# Bases are pinned by digest (reproducible; bump deliberately). Python 3.12 is a uv-managed standalone build,
# not the Ubuntu 22.04 apt package (which only ships 3.11 as a release candidate).
ARG CUDA_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.5@sha256:7bff3c3776ec467fc1437960f2c469d8beb30f536a6465a3350c647ccd260ec2

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- builder: toolchain + deps, never shipped
FROM ${CUDA_BASE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=8 UV_CONCURRENT_DOWNLOADS=4
RUN uv python install 3.12 && uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /src
# Hashed lock: torch 2.6.0 + triton 3.2.0 + CUDA libs resolved together (see requirements/README.md).
COPY requirements/lock-gpu-linux-py312.txt requirements/
RUN uv pip install --require-hashes -r requirements/lock-gpu-linux-py312.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --no-deps .

# ---------------------------------------------------------------- runtime: no uv, no pip cache, no source tree
FROM ${CUDA_BASE} AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH HF_HOME=/app/.cache/huggingface
# gcc + libc headers: Triton JIT-compiles a small C stub at first kernel launch.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
# Numeric UID so Kubernetes runAsNonRoot can verify it. Weights, data and secrets are mounted/pulled at run time.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin forge \
    && mkdir -p /app/runs /app/data /app/.cache && chown -R 10001:10001 /app
WORKDIR /app
# The worker validates TrainingJob specs against this schema (not packaged in the wheel).
COPY spec/jobspec.v1alpha1.schema.json /app/spec/jobspec.v1alpha1.schema.json
ENV TINYFORGE_JOBSPEC_SCHEMA=/app/spec/jobspec.v1alpha1.schema.json
USER 10001
EXPOSE 8000
# `serve --host 0.0.0.0` refuses to start without TLS (--ssl-certfile/--ssl-keyfile) and TINYFORGE_API_TOKEN, unless
# --insecure-http is given deliberately (e.g. TLS terminated by an ingress). The probe works for either mode (loopback only).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import ssl,urllib.request as u;c=ssl._create_unverified_context();exec('try:\n u.urlopen(\"http://127.0.0.1:8000/healthz\",timeout=4)\nexcept Exception:\n u.urlopen(\"https://127.0.0.1:8000/healthz\",timeout=4,context=c)')"
ENTRYPOINT ["tinyforge"]
CMD ["serve", "--host", "0.0.0.0"]
