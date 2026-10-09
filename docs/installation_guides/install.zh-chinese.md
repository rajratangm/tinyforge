# 在 Linux、Windows 和 Colab 上安装并使用 tinyforge

各平台上的命令相同,只有 shell 语法不同。每一步都并排给出 **Linux / macOS (bash)** 和
**Windows (PowerShell)** 两种写法。

## 各平台测试情况

| 平台 | 状态 |
|---|---|
| **Linux + NVIDIA GPU**(Google Colab、Ubuntu、Tesla T4 16 GB、Python 3.13) | 基于已发布的软件包完整跑通:安装、`doctor`、`memory`、数据工具、从零开始的流水线、LoRA 微调、作业规范 worker、服务器(OpenAI API、密钥防护、指标)、GGUF 导出以及 llama.cpp 服务。参见 `notebooks/colab_smoke_test.ipynb`。 |
| **Linux,仅 CPU**(Docker 和 GitHub Actions,Python 3.10 和 3.12) | 完整测试套件在 CI 中通过。 |
| **Windows 11 + NVIDIA GPU**(RTX 3050 Ti 4 GB,Python 3.12) | 主要开发机器;README 中"Verified results"的所有内容均在此运行。 |
| **Linux,2 块 NVIDIA GPU**(Kaggle 2x T4) | 已验证数据并行 LoRA:NCCL 可用,各副本一致,恢复训练可用;在小型任务上吞吐量仅为单 GPU 的 1.2x-1.4x。 |
| 多节点、带 GPU 的 Kubernetes、AMD (ROCm)、Apple silicon | **未测试。** Kubernetes chart 和 operator 相关部分仅在纯 CPU 的 `kind` 集群上验证过。 |

## 1. 前置条件

- Python 3.10 或更高版本(已在 3.10、3.12 和 3.13 上测试)。
- 使用 GPU 时:需要驱动正常的 NVIDIA GPU(`nvidia-smi` 必须可用)。**不需要**安装 CUDA 工具包;
  PyTorch 的 wheel 自带 CUDA 运行时。
- Windows:从 python.org 安装 Python(勾选 "Add to PATH")。Linux:`sudo apt install python3 python3-venv`
  (Debian/Ubuntu),或使用你的发行版的对应命令。

## 2. 创建环境并安装

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

可选扩展(Extras):`[finetune]` LoRA/QLoRA,`[docs]` PDF/Word/Excel 导入,`[bench]` 标准基准测试,
`[all]` 全部。如需从源码安装:在克隆的仓库中运行 `pip install -e ".[dev]"`。

检查安装结果:

```
tinyforge --version
tinyforge doctor        # hardware, what is installed, and the exact command for anything missing
```

在 Google Colab 和 Kaggle 上,torch 已预装:跳过 torch 那一行。(它们预装的 `torchao` 对
`peft` 来说版本太旧;tinyforge 会在自己的进程中自动将其隐藏。设置 `TINYFORGE_KEEP_TORCHAO=1` 可关闭此行为。)

## 3. 首次运行

以下命令在两种系统上的用法相同。

```
tinyforge pipeline --preset micro --steps 300       # from scratch: data -> train -> eval gates (about 1-3 min on a GPU)
tinyforge generate "ROMEO:" --int8
tinyforge ft pipeline --steps 100                    # fine-tune SmolLM2-360M with LoRA (downloads about 700 MB)
tinyforge ft generate "Explain hash tables" --base   # the original model; drop --base for the tuned one
```

输出会写入当前目录下的 `runs/` 和 `data/`(`runs/micro`、`runs/ft` 等)。在 Windows 上,相同的
路径写作 `runs\micro\best.pt`;参数中两种斜杠风格均可使用。

## 4. 部署模型并调用

服务器绑定到 localhost。身份验证默认开启,因此要么设置令牌(token),要么仅在本地实验时
将其关闭。

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

`/v1/chat/completions` 端点兼容 OpenAI,因此现有的 OpenAI 客户端库只需将 `base_url`
指向 `http://127.0.0.1:8000/v1` 即可使用。包含疑似凭据字符串的提示词会被拒绝,并返回 400。

若要监听 localhost 以外的地址,需要 TLS(`--ssl-certfile`、`--ssl-keyfile`)和令牌;请参阅
`docs/networking.md`。

## 5. 可选组件

| 你想要 | Linux / macOS | Windows |
|---|---|---|
| 将文档(PDF、Word、Excel)作为训练数据 | `pip install "tinyforge[docs]"` | 相同 |
| 标准基准测试(`bench run`) | `pip install "tinyforge[bench]"` | 相同 |
| 4 位(QLoRA) | 已包含在 `[finetune]` 中(bitsandbytes) | 已包含;需要较新的 bitsandbytes wheel |
| 融合 Triton 内核 | `pip install triton`(通常随 torch 一起安装) | `pip install triton-windows` |
| GGUF 导出和 llama.cpp 引擎 | 自行编译或下载 llama.cpp,然后 `export TINYFORGE_LLAMACPP_SRC=~/llama.cpp TINYFORGE_LLAMACPP_BIN=~/llama.cpp/build/bin TINYFORGE_LLAMA_SERVER=~/llama.cpp/build/bin/llama-server` | 下载 llama.cpp 发行版,然后 `$env:TINYFORGE_LLAMACPP_SRC="C:\llama.cpp"` 等(同样是这三个变量) |
| 训练超出显存(VRAM)的模型(`backend: soup`) | 在**独立的** venv 中安装 `soup-cli`,然后 `export TINYFORGE_SOUP_BIN=/path/to/soup` | 相同,`$env:TINYFORGE_SOUP_BIN="C:\path\soup.exe"` |

在 Linux 上编译 llama.cpp(导出和量化只需 CPU;若要用 GPU 提供服务,请添加 `-DGGML_CUDA=ON`):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DLLAMA_CURL=OFF && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server
pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt   # use a separate venv: it pins protobuf
```

然后运行:`tinyforge export gguf --base <local HF model folder> --adapter runs/ft/best --quant q8_0`,以及
`tinyforge serve --engine llamacpp --gguf out/gguf/base-q8_0.gguf --lora-gguf out/gguf/adapter-f16.gguf`。

## 6. 根据规范文件运行训练作业(Kubernetes 代理所运行的内容)

```
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1 --dry-run   # validate, change nothing
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1
```

退出码列在 `spec/worker-contract.md` 中(例如 75 表示"请重试")。

## 7. Linux 与 Windows 的差异

- **GPU 精度。** 早于 Ampere 架构的 GPU(计算能力低于 8.0,例如 T4)没有原生 bf16;
  tinyforge 会以 fp16 加损失缩放(loss scaling)来训练它们。Ampere 及更新的架构使用 bf16。
- **多 GPU。** `tinyforge ft train --gpus 2`(或作业规范中的 `resources: {gpus: 2}`)可在
  单机上以数据并行方式训练 LoRA:每块 GPU 在各自完整的 batch 上训练(因此有效 batch 是原来的 N 倍;`--split-batch` 则是将一个 batch 拆分),并在每一步对体积很小的 LoRA 梯度取平均。该功能
  需要 Linux(Windows 上没有 NCCL)。已通过 CPU 上真实的 2 进程运行以及 Kaggle 2x T4 机器
  (`notebooks/kaggle_multi_gpu.ipynb`)测试:各 GPU 保持一致,恢复训练可用,但在小型 360M 任务上吞吐量仅为
  单 GPU 的 1.2x-1.4x,而非 2x。计算占主导的更大模型应该扩展得更好;这一点尚未测试。不支持:将单个模型分片到多块 GPU 上(FSDP)、
  多节点,以及多于一块 GPU 的 `soup` 后端。
- **Triton、JAX、vLLM、DeepSpeed。** 以 Linux 为先。Windows 需要 `triton-windows`;JAX 在 Windows 上没有原生 GPU 支持。
- **WSL2。** 按 Linux 使用:安装 NVIDIA 的 Windows 驱动(不要在 WSL 内安装 Linux 驱动),然后在 WSL shell 中
  按照 Linux 一栏操作。
- **路径与引号。** 含空格的内容请加引号。在 PowerShell 中请调用 `curl.exe`(单独的 `curl` 是
  `Invoke-WebRequest` 的别名),并按上文所示对 JSON 引号进行转义。
- **笔记本电脑。** Windows 笔记本在空闲几分钟后会进入睡眠,这会中断长时间的训练;请更改电源计划,
  或在训练期间保持机器唤醒。
- **Docker。** 在 Linux 上,`docker run --gpus all` 需要 NVIDIA Container Toolkit;在 Windows 上,它随
  Docker Desktop 的 WSL2 后端一起提供。

## 8. 故障排除

| 症状 | 解决办法 |
|---|---|
| `tinyforge: This command needs 'torch'...` | 使用第 2 节中的那一行安装 torch,然后重新运行 `tinyforge doctor` |
| `incompatible version of torchao`(旧版 tinyforge) | 升级 tinyforge,或运行 `pip uninstall -y torchao` |
| `No checkpoint at runs/default/best.pt` | 先训练(`tinyforge pipeline`),或传入 `--ckpt runs/<preset>/best.pt` |
| `cuda_available False` 但你有 GPU | `nvidia-smi` 必须可用;安装 CUDA 版本的 torch(`cu124` 索引,而不是 `cpu`) |
| GPU 显存不足 | `tinyforge plan` / `tinyforge ft plan` 会说明能放下什么;降低 `--max-len`,使用 4 位,或参阅 `tinyforge memory` |
| `API token not configured`(HTTP 503) | 在 `serve` 之前设置 `TINYFORGE_API_TOKEN`(本地测试可设置 `TINYFORGE_AUTH=off`) |
