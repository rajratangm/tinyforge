# Instalar y usar tinyforge en Linux, Windows y Colab

Los comandos son los mismos en todas partes; solo cambia la sintaxis del shell. Cada paso muestra **Linux / macOS (bash)** y
**Windows (PowerShell)** lado a lado.

## Qué se ha probado y dónde

| Plataforma | Estado |
|---|---|
| **Linux + GPU NVIDIA** (Google Colab, Ubuntu, Tesla T4 de 16 GB, Python 3.13) | Se ejecutó de principio a fin desde el paquete publicado: instalación, `doctor`, `memory`, herramientas de datos, pipeline desde cero, ajuste fino con LoRA, worker con especificación de trabajo, servidor (API de OpenAI, protección contra secretos, métricas), exportación a GGUF y servicio con llama.cpp. Véase `notebooks/colab_smoke_test.ipynb`. |
| **Linux, solo CPU** (Docker y GitHub Actions, Python 3.10 y 3.12) | La suite de pruebas completa pasa en CI. |
| **Windows 11 + GPU NVIDIA** (RTX 3050 Ti de 4 GB, Python 3.12) | La máquina principal de desarrollo; todo lo que aparece en los "Verified results" del README se ejecutó aquí. |
| **Linux, 2 GPU NVIDIA** (Kaggle 2x T4) | LoRA con paralelismo de datos verificado: NCCL funciona, las réplicas son idénticas y la reanudación funciona; el rendimiento fue solo de 1.2x-1.4x respecto a una GPU en un trabajo pequeño. |
| Multinodo, Kubernetes con GPU, AMD (ROCm), Apple silicon | **No probado.** El chart de Kubernetes y los componentes del operador se verificaron únicamente en un clúster `kind` solo con CPU. |

## 1. Requisitos previos

- Python 3.10 o superior (probado en 3.10, 3.12 y 3.13).
- Para usar GPU: una GPU NVIDIA con un controlador que funcione (`nvidia-smi` debe funcionar). **No** necesitas el toolkit de CUDA;
  los wheels de PyTorch incluyen su propio runtime de CUDA.
- Windows: instala Python desde python.org (marca "Add to PATH"). Linux: `sudo apt install python3 python3-venv`
  (Debian/Ubuntu) o el equivalente de tu distribución.

## 2. Crear un entorno e instalar

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

Extras: `[finetune]` LoRA/QLoRA, `[docs]` ingesta de PDF/Word/Excel, `[bench]` benchmarks estándar,
`[all]` todo. Para instalar desde el código fuente: `pip install -e ".[dev]"` desde un clon.

Comprueba el resultado:

```
tinyforge --version
tinyforge doctor        # hardware, what is installed, and the exact command for anything missing
```

En Google Colab y Kaggle, torch ya viene preinstalado: omite la línea de torch. (El `torchao` preinstalado allí es demasiado antiguo para
`peft`; tinyforge lo oculta automáticamente de su propio proceso. Define `TINYFORGE_KEEP_TORCHAO=1` para desactivarlo.)

## 3. Primera ejecución

Esto funciona igual en ambos sistemas.

```
tinyforge pipeline --preset micro --steps 300       # from scratch: data -> train -> eval gates (about 1-3 min on a GPU)
tinyforge generate "ROMEO:" --int8
tinyforge ft pipeline --steps 100                    # fine-tune SmolLM2-360M with LoRA (downloads about 700 MB)
tinyforge ft generate "Explain hash tables" --base   # the original model; drop --base for the tuned one
```

Las salidas van a `runs/` y `data/` dentro del directorio actual (`runs/micro`, `runs/ft`, ...). En Windows las mismas
rutas son `runs\micro\best.pt`; en los argumentos sirve cualquiera de los dos estilos de barra.

## 4. Servir el modelo y llamarlo

El servidor escucha en localhost. La autenticación está activada por defecto, así que o bien defines un token o, solo para
experimentos locales, la desactivas.

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

El endpoint `/v1/chat/completions` es compatible con OpenAI, por lo que las bibliotecas cliente de OpenAI existentes funcionan
apuntando `base_url` a `http://127.0.0.1:8000/v1`. Un prompt que contenga una cadena con aspecto de credencial se rechaza con un 400.

Para escuchar en cualquier dirección distinta de localhost necesitas TLS (`--ssl-certfile`, `--ssl-keyfile`) y un token; consulta
`docs/networking.md`.

## 5. Componentes opcionales

| Quieres | Linux / macOS | Windows |
|---|---|---|
| Documentos (PDF, Word, Excel) como datos de entrenamiento | `pip install "tinyforge[docs]"` | igual |
| Benchmarks estándar (`bench run`) | `pip install "tinyforge[bench]"` | igual |
| 4 bits (QLoRA) | incluido en `[finetune]` (bitsandbytes) | incluido; requiere un wheel reciente de bitsandbytes |
| Kernel Triton fusionado | `pip install triton` (normalmente se instala con torch) | `pip install triton-windows` |
| Exportación a GGUF y el motor llama.cpp | compila o descarga llama.cpp y luego `export TINYFORGE_LLAMACPP_SRC=~/llama.cpp TINYFORGE_LLAMACPP_BIN=~/llama.cpp/build/bin TINYFORGE_LLAMA_SERVER=~/llama.cpp/build/bin/llama-server` | descarga una versión de llama.cpp y luego `$env:TINYFORGE_LLAMACPP_SRC="C:\llama.cpp"` etc. (las mismas tres variables) |
| Entrenar modelos más grandes que la VRAM (`backend: soup`) | instala `soup-cli` en su **propio** venv y luego `export TINYFORGE_SOUP_BIN=/path/to/soup` | igual, `$env:TINYFORGE_SOUP_BIN="C:\path\soup.exe"` |

Compilar llama.cpp en Linux (la CPU basta para exportar y cuantizar; añade `-DGGML_CUDA=ON` para servir con GPU):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DLLAMA_CURL=OFF && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server
pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt   # use a separate venv: it pins protobuf
```

Después: `tinyforge export gguf --base <local HF model folder> --adapter runs/ft/best --quant q8_0` y
`tinyforge serve --engine llamacpp --gguf out/gguf/base-q8_0.gguf --lora-gguf out/gguf/adapter-f16.gguf`.

## 6. Ejecutar un trabajo de entrenamiento desde una especificación (lo que ejecuta el agente de Kubernetes)

```
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1 --dry-run   # validate, change nothing
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1
```

Los códigos de salida se enumeran en `spec/worker-contract.md` (por ejemplo, 75 significa "reinténtame").

## 7. Diferencias entre Linux y Windows que conviene conocer

- **Precisión en GPU.** Las GPU anteriores a Ampere (capacidad de cómputo inferior a 8.0, como la T4) no tienen bf16 nativo;
  tinyforge las entrena en fp16 con escalado de pérdida (loss scaling). Ampere y posteriores usan bf16.
- **Multi-GPU.** `tinyforge ft train --gpus 2` (o `resources: {gpus: 2}` en una especificación de trabajo) entrena LoRA con paralelismo de datos en
  una sola máquina: cada GPU entrena con su propio batch completo (por lo que el batch efectivo es N veces mayor; `--split-batch` divide un único batch) y los pequeños gradientes de LoRA se promedian en cada paso. Requiere
  Linux (NCCL no existe en Windows). Probado con ejecuciones reales de 2 procesos en CPU y en una máquina Kaggle con 2x T4
  (`notebooks/kaggle_multi_gpu.ipynb`): las GPU se mantienen idénticas y la reanudación funciona, pero el rendimiento fue solo de 1.2x-1.4x respecto a
  una GPU en un trabajo pequeño de 360M, no de 2x. Los modelos más grandes, donde domina el cómputo, deberían escalar mejor; eso no se ha probado. No disponible: repartir un modelo entre varias GPU (FSDP),
  multinodo, y el backend `soup` con más de una GPU.
- **Triton, JAX, vLLM, DeepSpeed.** Primero Linux. Windows necesita `triton-windows`; JAX no tiene soporte nativo de GPU en Windows.
- **WSL2.** Funciona como Linux: instala el controlador NVIDIA de Windows (no un controlador de Linux dentro de WSL) y luego sigue la
  columna de Linux dentro del shell de WSL.
- **Rutas y comillas.** Usa comillas alrededor de cualquier cosa con espacios. En PowerShell, llama a `curl.exe` (el `curl` a secas es un
  alias de `Invoke-WebRequest`) y escapa las comillas del JSON como se muestra arriba.
- **Portátiles.** Los portátiles con Windows se suspenden tras unos minutos de inactividad, lo que detiene un entrenamiento largo; cambia el plan de energía
  o mantén el equipo activo mientras entrenas.
- **Docker.** `docker run --gpus all` requiere el NVIDIA Container Toolkit en Linux; en Windows viene con el backend WSL2 de
  Docker Desktop.

## 8. Solución de problemas

| Síntoma | Solución |
|---|---|
| `tinyforge: This command needs 'torch'...` | instala torch con la línea de la sección 2 y vuelve a ejecutar `tinyforge doctor` |
| `incompatible version of torchao` (tinyforge antiguo) | actualiza tinyforge, o `pip uninstall -y torchao` |
| `No checkpoint at runs/default/best.pt` | entrena primero (`tinyforge pipeline`), o pasa `--ckpt runs/<preset>/best.pt` |
| `cuda_available False` pero tienes una GPU | `nvidia-smi` debe funcionar; instala la versión CUDA de torch (índice `cu124`, no `cpu`) |
| Sin memoria de GPU | `tinyforge plan` / `tinyforge ft plan` indican qué cabe; reduce `--max-len`, usa 4 bits, o consulta `tinyforge memory` |
| `API token not configured` (HTTP 503) | define `TINYFORGE_API_TOKEN` (o `TINYFORGE_AUTH=off` para pruebas locales) antes de `serve` |
