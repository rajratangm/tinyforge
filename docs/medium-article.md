# Teach an AI model new tricks on the laptop you already own (and know if it worked)

*tinyforge: fine-tune, check and serve small language models on your own GPU. Windows and Linux, one `pip install`. This guide works for a complete beginner and for an engineer who wants the flags, the exit codes and the honest numbers.*

---

## The 30-second version

- **What it is:** a free, open-source command-line tool (Apache-2.0) that takes an existing AI model and teaches it your task, using the graphics card you already have, even a small 4 GB laptop one.
- **What makes it different:** it does not just say "training finished". It compares your tuned model with the original on data the model never saw, and **fails loudly** if the result is not better.
- **How to get it:** `pip install tinyforge` (current version **0.1.2**).
- **Where it was tested:** a Windows laptop (RTX 3050 Ti, 4 GB), a free Google Colab T4, and a Kaggle machine with two T4s. Not tested: AMD/Apple GPUs, Kubernetes with GPUs, multi-machine training.

Repo: https://github.com/rajratangm/tinyforge

---

## Part 1. For everyone: what is "fine-tuning"?

Think of a big AI model as a student who has read half the internet. They know a little about everything, but they do not know *your* thing: your company's way of writing, your database, your support answers.

**Fine-tuning** is a short tutoring session. You show the student a few thousand examples of your task, and they get better at exactly that.

The usual problem: tutoring a big model needs a big, expensive GPU. tinyforge uses clever tricks (small add-on layers called **LoRA**, and squeezing the model to 4-bit numbers, called **QLoRA**) so the same tutoring fits on a small card. You do not have to choose the tricks. The tool looks at your machine and chooses for you.

The second problem: many tools say "done!" even when the model got no better. tinyforge **tests the student before and after**, and tells you the truth.

---

## Part 2. Your first run (about 15 minutes, copy and paste)

### What you need

- Python 3.10 or newer.
- An NVIDIA GPU is best. Without one it still installs, but training will be very slow. A free Google Colab GPU works too.

### Step 1. Make a clean workspace

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "tinyforge[finetune]"
```

**Linux / macOS**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # use /cpu if you have no NVIDIA GPU
pip install "tinyforge[finetune]"
```

Why install `torch` yourself first? PyTorch comes in different builds (for different GPUs). Only you know which one your computer needs, so tinyforge leaves that choice to you instead of guessing wrong.

### Step 2. Ask the tool about your computer

```bash
tinyforge doctor
```

You get a table: your GPU, its memory, which features it supports, and a list of optional parts that are `ok` or `missing`. For every missing part it prints the exact command to fix it. If you see `torch_version: None`, you skipped the torch install in step 1.

### Step 3. Teach a model

```bash
tinyforge ft pipeline --steps 100
```

This one command does everything, in order:

1. Checks your hardware.
2. Downloads a small training dataset, removes duplicates, splits it into "practice" and "exam" parts so the exam is never seen during training, and scans for passwords or personal data.
3. Plans the memory use so you do not wait twenty minutes for an out-of-memory crash.
4. Trains (the default model is the small `SmolLM2-360M-Instruct`).
5. Grades the tuned model against the original on the exam part.
6. Merges the result into a usable model.

If any check fails, the command exits with an error code and a message that says what went wrong. The tool does not print a cheerful green tick over a bad result.

### Step 4. Talk to your model

```bash
tinyforge ft generate "Explain hash tables in two sentences"
```

To hear the **original** model for comparison, add `--base`:

```bash
tinyforge ft generate "Explain hash tables in two sentences" --base
```

### Step 5. Open the web page and API

```bash
tinyforge serve
```

Open http://127.0.0.1:8000 in your browser. The same server speaks the OpenAI chat format, so existing tools and libraries can talk to it.

### If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `torch_version: None` in `doctor` | PyTorch is not installed | Run the `pip install torch ...` line from step 1 |
| A "missing component" message | An optional part is not installed | Copy the command the message prints |
| A plan that says the model does not fit | Your GPU is too small for that model | Pick a smaller `--base-model`, or see Part 4 |
| `WARN FE002: Only 1.8% improvement` | The tuned model is barely better | Not a bug. Your data did not teach much. Use data closer to your real task |

---

## Part 3. What results to expect (honest numbers)

On a free Colab T4, the default pipeline improved held-out loss from 1.314 to 1.271 (about 3%) with no forgetting of general text. A general chat dataset on an already-tuned small model mostly changes *style*, and the tool will say so when the gain is small.

Fine-tuning pays off on **narrow tasks**. I tuned models to write SQL for a table they had never seen, and scored them by *running the generated query* on the database (not by comparing text):

| Model | Tuned | Best prompted base model |
|---|---|---|
| Qwen2.5-3B, new table | **98%** correct | 70% |
| Llama-3.1-8B, new table (templated questions) | **100%** | 93% |
| Llama-3.1-8B, harder question shapes | 98% | 97% |

Look at the last row: on harder questions the tuned model was barely better than a well-written prompt. I left it in the README on purpose. These are single runs on small test sets (the raw JSON is in the repo's `benchmarks/` folder), so treat them as evidence, not proof.

Make your own score with a real task check:

```bash
tinyforge ft task-eval --task sql --n 100 --min-gain-pts 5
```

It exits non-zero unless the tuned model beats the base by at least 5 points.

---

## Part 4. For professionals: the full toolbox

### 4.1 Pick a method that fits the machine

```bash
tinyforge memory --params-b 8          # where would an 8B model's weights live (VRAM / RAM / NVMe), rough speed
tinyforge methods list                  # known fine-tuning methods and what has actually been verified
tinyforge methods suggest --params-b 8  # which methods fit this GPU and RAM, on which backend, and why
tinyforge ft plan                       # will the chosen preset fit, and with what settings (4-bit? checkpointing? batch?)
```

The speed estimates are uncalibrated. Treat tokens-per-second figures as rough.

### 4.2 Train with explicit settings

```bash
tinyforge ft train \
  --base-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --steps 300 --max-len 512 --batch-size 4 --grad-accum 4 \
  --lr 0.0002 --lora-r 16 --quant auto \
  --data-dir data/ft --run-dir runs/ft
tinyforge ft eval        # held-out loss, forgetting check, sample generations, merge check
tinyforge ft card        # writes a model card (README.md) from the run's own config and results
```

Training resumes automatically from `runs/ft/last.pt` and saves the best adapter to `runs/ft/best`. Add `--json` to any of these for machine-readable events, and use the exit code in CI: `pipeline` returns 1 if a gate fails.

### 4.3 Bring your own data, safely

```bash
tinyforge data tabular people.csv --out tab.jsonl --count 500     # CSV -> text-to-SQL pairs, kept only if the SQL executes
tinyforge data ingest ./docs --out chunks.jsonl                    # documents -> cleaned, chunked, de-duplicated text
tinyforge data pairs chunks.jsonl --teacher-url http://127.0.0.1:8000/v1 --out data/pairs   # chunks -> grounded Q&A via a "teacher" model server you point it at; split by source file BEFORE generation
tinyforge data pii-scan tab.jsonl --redact-to clean.jsonl          # exits 1 if it finds a secret
```

Splitting by source file *before* the question-writer runs prevents the usual leak where the same document appears in both training and exam data.

### 4.4 A reproducible job file

For repeatable runs (and for CI), describe the job in YAML:

```yaml
# job.yaml  (see spec/examples/ in the repo)
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata:
  name: sql-llama31-8b-streamed
spec:
  backend: soup            # layers stream through the GPU one at a time
  method: qlora            # 4-bit base
  model:
    base: NousResearch/Meta-Llama-3.1-8B-Instruct
  data:
    source: b-mc2/sql-create-context
    format: sql-create-context
    limit: 2000
    maxLen: 512
  hyperparameters: {maxSteps: 100, batchSize: 1, gradAccum: 4}
  resources: {gpus: 1}
  export: [adapter]
```

```bash
tinyforge worker run --spec job.yaml --out jobs/sql-8b
```

This is how a model **bigger than your GPU** is trained: an 8B model needs about 5 GB for 4-bit weights alone, so the optional layer-streaming backend (Soup, installed in its own environment) passes layers through the GPU one at a time. A 200-step run trained Llama-3.1-8B in about 9.5 minutes on the 4 GB laptop. It needs roughly 4 GB of free system RAM. Caveat: that backend is beta upstream and I verified it only on Windows.

### 4.5 More than one GPU

```bash
tinyforge ft train --steps 60 --gpus 2
```

Or `resources: {gpus: 2}` in a job file. This is data-parallel LoRA: each GPU trains on its own full batch and the small adapter gradients are averaged every step (`--split-batch` splits one batch across GPUs instead). At the end the tool compares the weights on every GPU and reports `ddp_max_param_divergence`. On a Kaggle 2×T4 it was exactly `0.0`, and resuming from a checkpoint stayed in lockstep.

**Honest speedup:** across four runs, two T4s gave **1.2× to 1.4×** the throughput of one, not 2×. Waiting between GPUs is measured and low, so the cause is something else (probably the shared host with four virtual CPUs). I will not claim near-linear scaling until a controlled test says so. It does not shard one big model across GPUs, and there is no multi-node training.

### 4.6 Check against standard benchmarks

```bash
tinyforge bench list      # what each benchmark measures and how it runs
tinyforge bench suggest   # sizes and picks benchmarks for your GPU, RAM and time budget
tinyforge bench run       # lm-evaluation-harness, or the built-in SQL execution check
```

Note: the built-in SQL check is the one I have run for real. I have **not** yet run the lm-evaluation-harness path end to end.

### 4.7 Export and serve

```bash
tinyforge export gguf --base ./SmolLM2-360M-Instruct --adapter runs/ft/best --quant q8_0
tinyforge serve --engine llamacpp \
  --gguf out/gguf/base-q8_0.gguf --lora-gguf out/gguf/adapter-f16.gguf
```

`q8_0` is the safe default. In my test `q4_k_m` lost about 5 points of exact match. Export needs the llama.cpp tools (`doctor` tells you how to get them).

Calling the OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $TINYFORGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

Safety defaults worth knowing:

- Binding to anything other than localhost needs TLS (or an explicit `--insecure-http`) **and** `TINYFORGE_API_TOKEN` set. mTLS is available with `--ssl-ca-certs` and `--client-cert-required`.
- If a prompt contains something that looks like a credential, the server refuses it with `secret_in_prompt` instead of sending it to the model.
- A model with random weights, or one with no chat template, is rejected with a specific diagnostic code rather than quietly served.

---

## Part 5. The part most projects skip: what broke when I tested on other machines

I built everything on one Windows laptop, and "works on my machine" proves almost nothing. So I ran the **published package** on a free Colab T4 (Linux) and a Kaggle 2×T4 box. These are the bugs that surfaced, all fixed in 0.1.1:

1. **The README quickstart failed.** `pipeline` saved to one folder and `generate` looked in another. Now it finds the newest checkpoint and explains what to do if there is none.
2. **Colab's preinstalled `torchao` crashed fine-tuning.** `peft` refuses versions older than 0.16 and Colab ships 0.10. tinyforge now hides a too-old `torchao` from its own process (opt out with `TINYFORGE_KEEP_TORCHAO=1`).
3. **A broken TensorFlow crashed every command.** Colab ships TensorFlow, and installing the llama.cpp converter downgrades `protobuf` under it. tinyforge is PyTorch-only, so it now turns TensorFlow, Flax and JAX off for `transformers`.
4. **The T4 ran about 2.7× slower than it should.** PyTorch says a T4 supports bf16, but only by emulating it. tinyforge now checks for real support (compute capability 8 or higher) and uses fp16 otherwise: about 1,100 to 2,960 tokens/s, and peak memory from 9.6 GB to 2.4 GB.
5. **That fix created a new bug.** In fp16, held-out loss sometimes came out `NaN`. Evaluation now recomputes an overflowing batch in bf16 and tells you (diagnostic `FT007`).
6. **My first multi-GPU design under-used the cards.** I split one fixed batch across GPUs, so each got half. The default is now a full batch per GPU.
7. **My own test notebook lied.** It showed a failing test suite as passing because a `| tail` hid the exit code.

None of these appeared on my laptop. All would have hit a real user in the first five minutes. Version 0.1.2 only corrected the README (it wrongly claimed multi-GPU was untested).

---

## Part 6. What it does not do (yet)

- **No sharded training** of one model across several GPUs (FSDP), and **no multi-node training**.
- **Kubernetes with GPUs is untested.** The Helm chart and job-spec pieces were verified only on a CPU-only `kind` cluster.
- **DPO / preference tuning** is listed but has never been run.
- **NVIDIA only.** No AMD or Apple GPU support.
- **Gains depend on the task:** big on narrow, templated tasks, modest on general chat, close to nothing on the harder question shapes I tried.
- **Speed estimates are rough**, and Windows multi-GPU is development-only (no NCCL on Windows).

---

## Cheat sheet

| I want to... | Command |
|---|---|
| See if my computer is ready | `tinyforge doctor` |
| See if a model fits my GPU | `tinyforge ft plan` / `tinyforge memory --params-b 8` |
| Train and check, all in one | `tinyforge ft pipeline --steps 100` |
| Chat with my model | `tinyforge ft generate "..."` (add `--base` for the original) |
| Prove it beat the base model | `tinyforge ft task-eval --task sql --min-gain-pts 5` |
| Clean my data of secrets | `tinyforge data pii-scan in.jsonl --redact-to out.jsonl` |
| Use two GPUs | `tinyforge ft train --gpus 2` |
| Run from a job file | `tinyforge worker run --spec job.yaml --out jobs/x` |
| Export for llama.cpp | `tinyforge export gguf --base ... --adapter ... --quant q8_0` |
| Start the web page and API | `tinyforge serve` |

## Try it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "tinyforge[finetune]"
tinyforge doctor
tinyforge ft pipeline --steps 100
```

There are ready-made test notebooks for Google Colab and Kaggle in `notebooks/`, and a step-by-step install guide in `docs/install.md`. If it breaks on your hardware, that is exactly what I want to know: https://github.com/rajratangm/tinyforge/issues

*tinyforge is Apache-2.0 licensed and early alpha (0.1.x). The numbers above come from single runs on small test sets; the raw JSON for each is committed in `benchmarks/`.*
