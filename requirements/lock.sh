#!/usr/bin/env sh
# Regenerate the hashed Linux locks. Needs uv (https://docs.astral.sh/uv/).
set -eu
cd "$(dirname "$0")/.."
GPU_PKGS="torch triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12"
NE=""; for p in $GPU_PKGS; do NE="$NE --no-emit-package $p"; done
for v in 3.10 3.12; do
  t=$(echo $v | tr -d .)
  common="pyproject.toml --extra dev --extra finetune --python-version $v --python-platform x86_64-unknown-linux-gnu --generate-hashes -c requirements/torch-constraint.txt -q"
  uv pip compile $common -o requirements/lock-gpu-linux-py$t.txt
  uv pip compile $common $NE -o requirements/lock-cpu-linux-py$t.txt
done
