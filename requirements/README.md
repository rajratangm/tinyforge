# Locked dependencies

Hashed, fully resolved, Linux x86_64, generated with `uv pip compile` (see the header of each file).
`torch` is pinned separately (`torch-constraint.txt`) because the wheel comes from a CUDA-specific index.

| File | Use |
|---|---|
| `lock-gpu-linux-pyXY.txt` | Everything incl. torch 2.6.0 + triton 3.2.0 + CUDA 12.4 libs, resolved together (Docker/GPU nodes) |
| `lock-cpu-linux-pyXY.txt` | Same minus torch/triton/nvidia-*; CI installs CPU torch first |

Install: `pip install --require-hashes -r requirements/lock-gpu-linux-py312.txt && pip install --no-deps -e .`

Regenerate (after editing pyproject.toml): `make lock`. Not covered: Windows/macOS (dev box uses the venv directly).
