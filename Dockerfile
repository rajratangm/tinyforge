# syntax=docker/dockerfile:1
# Bases are pinned by digest (reproducible; bump deliberately). Python 3.12 is a uv-managed standalone build,
# not the Ubuntu 22.04 apt package (which only ships 3.11 as a release candidate).
ARG CUDA_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.5@sha256:7bff3c3776ec467fc1437960f2c469d8beb30f536a6465a3350c647ccd260ec2

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- builder: toolchain + deps, never shipped
FROM ${CUDA_BASE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1
RUN uv python install 3.12 && uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu124
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install ".[triton,finetune]"

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
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
ENTRYPOINT ["tinyforge"]
CMD ["serve", "--host", "0.0.0.0"]
