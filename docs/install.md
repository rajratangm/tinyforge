# Install and use tinyforge on Linux, Windows and Colab

The commands are the same everywhere; only the shell syntax differs. Each step shows **Linux / macOS (bash)** and
**Windows (PowerShell)** side by side.

## What has been tested where

| Platform | Status |
|---|---|
| **Linux + NVIDIA GPU** (Google Colab, Ubuntu, Tesla T4 16 GB, Python 3.13) | Ran end to end from the published package: install, `doctor`, `memory`, data tools, from-scratch pipeline, LoRA fine-tune, job-spec worker, server (OpenAI API, secret guardrail, metrics), GGUF export and llama.cpp serving. See `notebooks/colab_smoke_test.ipynb`. |
| **Linux, CPU only** (Docker and GitHub Actions, Python 3.10 and 3.12) | Full test suite passes in CI. |
| **Windows 11 + NVIDIA GPU** (RTX 3050 Ti 4 GB, Python 3.12) | The main development machine; everything in the README "Verified results" was run here. |
| Multi-GPU, multi-node, Kubernetes with GPUs, AMD (ROCm), Apple silicon | **Not tested.** The Kubernetes chart and operator pieces were verified on a CPU-only `kind` cluster only. |

## 1. Prerequisites

- Python 3.10 or newer (tested on 3.10, 3.12 and 3.13).
- For GPU use: an NVIDIA GPU with a working driver (`nvidia-smi` must work). You do **not** need the CUDA toolkit;
  the PyTorch wheels bring their own CUDA runtime.
- Windows: install Python from python.org (tick "Add to PATH"). Linux: `sudo apt install python3 python3-venv`
  (Debian/Ubuntu) or your distribution's equivalent.

## 2. Create an environment and install

**Linux / macOS (bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU. For CPU only: .../whl/cpu
pip install "tinyforge[finetune]"
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU. For CPU only: .../whl/cpu
pip install "tinyforge[finetune]"
```

Extras: `[finetune]` LoRA/QLoRA, `[docs]` PDF/Word/Excel ingestion, `[bench]` standard benchmarks,
`[all]` everything. Installing from source instead: `pip install -e ".[dev]"` from a clone.

Check the result:

```
tinyforge --version
tinyforge doctor        # hardware, what is installed, and the exact command for anything missing
```

On Google Colab, torch is preinstalled: skip the torch line. Colab ships a `torchao` that `peft` rejects; if you
see "incompatible version of torchao", run `pip uninstall -y torchao`.

## 3. First run

These work the same on both systems.

```
tinyforge pipeline --preset micro --steps 300       # from scratch: data -> train -> eval gates (about 1-3 min on a GPU)
tinyforge generate "ROMEO:" --int8
tinyforge ft pipeline --steps 100                    # fine-tune SmolLM2-360M with LoRA (downloads about 700 MB)
tinyforge ft generate "Explain hash tables" --base   # the original model; drop --base for the tuned one
```

Outputs go to `runs/` and `data/` under the current directory (`runs/micro`, `runs/ft`, ...). On Windows the same
paths are `runs\micro\best.pt`; either slash style works in arguments.

## 4. Serve the model and call it

The server binds to localhost. Authentication is on by default, so either set a token or, for local experiments
only, turn it off.

**Linux / macOS (bash)**

```bash
export TINYFORGE_API_TOKEN="$(openssl rand -hex 24)"
tinyforge serve --port 8000                     # UI at http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $TINYFORGE_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
# local experiments only, no token:   TINYFORGE_AUTH=off tinyforge serve
```

**Windows (PowerShell)**

```powershell
$env:TINYFORGE_API_TOKEN = -join ((48..57 + 97..102) | Get-Random -Count 48 | ForEach-Object {[char]$_})
tinyforge serve --port 8000                     # UI at http://127.0.0.1:8000
curl.exe -s http://127.0.0.1:8000/healthz
curl.exe -s http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:TINYFORGE_API_TOKEN" -H "Content-Type: application/json" `
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":16}'
# local experiments only, no token:   $env:TINYFORGE_AUTH = "off"; tinyforge serve
```

The `/v1/chat/completions` endpoint is OpenAI-compatible, so existing OpenAI client libraries work by pointing
`base_url` at `http://127.0.0.1:8000/v1`. A prompt containing a credential-like string is refused with a 400.

To listen on anything other than localhost you need TLS (`--ssl-certfile`, `--ssl-keyfile`) and a token; see
`docs/networking.md`.

## 5. Optional pieces

| You want | Linux / macOS | Windows |
|---|---|---|
| Documents (PDF, Word, Excel) as training data | `pip install "tinyforge[docs]"` | same |
| Standard benchmarks (`bench run`) | `pip install "tinyforge[bench]"` | same |
| 4-bit (QLoRA) | included in `[finetune]` (bitsandbytes) | included; needs a recent bitsandbytes wheel |
| Fused Triton kernel | `pip install triton` (usually installed with torch) | `pip install triton-windows` |
| GGUF export and the llama.cpp engine | build or download llama.cpp, then `export TINYFORGE_LLAMACPP_SRC=~/llama.cpp TINYFORGE_LLAMACPP_BIN=~/llama.cpp/build/bin TINYFORGE_LLAMA_SERVER=~/llama.cpp/build/bin/llama-server` | download a llama.cpp release, then `$env:TINYFORGE_LLAMACPP_SRC="C:\llama.cpp"` etc. (same three variables) |
| Train models bigger than VRAM (`backend: soup`) | install `soup-cli` in its **own** venv, then `export TINYFORGE_SOUP_BIN=/path/to/soup` | same, `$env:TINYFORGE_SOUP_BIN="C:\path\soup.exe"` |

Building llama.cpp on Linux (CPU is enough for export and quantisation; add `-DGGML_CUDA=ON` for GPU serving):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DLLAMA_CURL=OFF && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server
pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt   # use a separate venv: it pins protobuf
```

Then: `tinyforge export gguf --base <local HF model folder> --adapter runs/ft/best --quant q8_0` and
`tinyforge serve --engine llamacpp --gguf out/gguf/base-q8_0.gguf --lora-gguf out/gguf/adapter-f16.gguf`.

## 6. Run a training job from a spec (what the Kubernetes agent runs)

```
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1 --dry-run   # validate, change nothing
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1
```

Exit codes are listed in `spec/worker-contract.md` (for example 75 means "retry me").

## 7. Linux and Windows differences worth knowing

- **GPU precision.** GPUs older than Ampere (compute capability below 8.0, such as the T4) have no native bf16;
  tinyforge trains them in fp16 with loss scaling. Ampere and newer use bf16.
- **Multi-GPU.** `tinyforge ft train --gpus 2` (or `resources: {gpus: 2}` in a job spec) trains LoRA data-parallel on
  one machine: each GPU processes its share of every batch and the small LoRA gradients are averaged each step. It
  needs Linux (NCCL does not exist on Windows). Tested with real 2-process runs on CPU; real-GPU NCCL runs are what
  `notebooks/kaggle_multi_gpu.ipynb` (Kaggle, 2x T4) measures. Not available: sharding one model across GPUs (FSDP),
  multi-node, and the `soup` backend with more than one GPU.
- **Triton, JAX, vLLM, DeepSpeed.** Linux first. Windows needs `triton-windows`; JAX has no native Windows GPU support.
- **WSL2.** Works as Linux: install the NVIDIA Windows driver (not a Linux driver inside WSL), then follow the Linux
  column inside the WSL shell.
- **Paths and quoting.** Use quotes around anything with spaces. In PowerShell, call `curl.exe` (plain `curl` is an
  alias of `Invoke-WebRequest`) and escape the JSON quotes as shown above.
- **Laptops.** Windows laptops sleep after a few idle minutes, which stops a long training run; change the power plan
  or keep the machine awake while training.
- **Docker.** `docker run --gpus all` needs the NVIDIA Container Toolkit on Linux; on Windows it comes with Docker
  Desktop's WSL2 backend.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `tinyforge: This command needs 'torch'...` | install torch with the line in section 2, then re-run `tinyforge doctor` |
| `incompatible version of torchao` | `pip uninstall -y torchao` (or `pip install -U torchao`) |
| `No checkpoint at runs/default/best.pt` | train first (`tinyforge pipeline`), or pass `--ckpt runs/<preset>/best.pt` |
| `cuda_available False` but you have a GPU | `nvidia-smi` must work; install the CUDA build of torch (`cu124` index, not `cpu`) |
| Out of GPU memory | `tinyforge plan` / `tinyforge ft plan` say what fits; lower `--max-len`, use 4-bit, or see `tinyforge memory` |
| `API token not configured` (HTTP 503) | set `TINYFORGE_API_TOKEN` (or `TINYFORGE_AUTH=off` for local tests) before `serve` |
