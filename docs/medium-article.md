# I built a tool that fine-tunes LLMs on a 4 GB laptop GPU, then tested it on Colab and Kaggle until it broke

*tinyforge: train, check, and serve small language models on the GPU you already own. Windows and Linux, one `pip install`.*

---

Most "fine-tune an LLM" tutorials quietly assume you have a 24 GB card or a cloud budget. I have a laptop with an RTX 3050 Ti and **4 GB of VRAM**. That is not enough to load an 8-billion-parameter model, let alone train one.

So I built **tinyforge**, an open-source tool whose goal is simple: *you point it at the hardware you have, and it figures out how to train something useful on it, and then tells you honestly whether the result is any good.*

This post is the story of what it does, how to use it, and the part most project write-ups skip: **what broke when I tested it on machines that weren't mine.**

```bash
pip install tinyforge
```

Repo: https://github.com/rajratangm/tinyforge

---

## What it does, in one sentence

tinyforge fine-tunes, evaluates, and serves small-to-mid LLMs on whatever NVIDIA GPU you have, and it checks that the tuned model is actually better than the base model instead of just reporting that training finished.

## Why not just use an existing trainer?

Existing trainers are excellent, but they mostly answer "how do I train?" and leave three questions to you:

1. **Will it fit on my GPU?** tinyforge probes your VRAM, RAM and disk, then plans: fp16 or 4-bit, gradient checkpointing, micro-batch size. If your GPU is too small, it says so *before* you wait twenty minutes for an out-of-memory crash.
2. **Did it actually get better?** Every run is judged against the base model on held-out data. A random-weights model, a model with no chat template, or a tuned model that is no better than the base all fail with a specific error code instead of a green tick.
3. **How do I use it afterwards?** One command exports to GGUF; another serves an OpenAI-compatible API with a secret-leak guardrail.

---

## Getting started (Windows and Linux)

The commands are identical on both systems. Only the shell syntax differs.

**Linux / macOS**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or /cpu
pip install "tinyforge[finetune]"
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "tinyforge[finetune]"
```

Then ask the tool what it thinks of your machine:

```bash
tinyforge doctor
```

```text
(abridged)
os                 Linux 6.6.122+
gpu_name           Tesla T4
vram_gb            14.56
compute_capability (7, 5)
...
Optional components
  finetune      ok
  llama.cpp     missing   -> download llama.cpp, set TINYFORGE_LLAMA_SERVER ...
```

`doctor` tells you what is missing and gives the exact command to fix it. No stack traces for a forgotten dependency.

`tinyforge memory` goes one step further: it measures your VRAM, RAM and disk bandwidth and says where a model's weights would live and roughly how fast it would run.

---

## Fine-tune a model in two commands

```bash
tinyforge ft pipeline --steps 100
tinyforge ft generate "Explain hash tables in two sentences"
```

The first command runs the whole chain: check hardware, prepare the data (de-duplicated, leak-free train/validation split, PII and secret scan), plan the memory, train a LoRA adapter, evaluate tuned versus base, and merge. It exits non-zero if a gate fails.

On a free Colab T4 one run improved held-out loss from 1.314 to 1.271 (about 3%) with no forgetting on general text. On another run of the same pipeline, with a weaker improvement, it printed this instead of celebrating:

```text
WARN  FE002: Only 1.8% improvement over base: the data may not teach anything new.
```

That is the point. A general instruction dataset on an already-tuned small model mostly changes style. The tool says so.

### Where fine-tuning really pays off: a narrow task

I tuned models to write SQL for a table they had never seen. Results, measured by *running the generated query* against the database, not by string matching:

| Model | Tuned | Best prompted base |
|---|---|---|
| Qwen2.5-3B, new table | **98%** execution accuracy | 70% |
| Llama-3.1-8B, new table (templated questions) | **100%** | 93% |
| Llama-3.1-8B, harder question shapes | 98% | 97% |

Read the last row carefully. On harder questions the tuned model was barely better than a well-prompted base. Fine-tuning mostly fixed the *output format*. I left that row in the README because a fine-tuning tool that only shows its wins is advertising, not engineering.

---

## Training a model bigger than your GPU

An 8B model in 4-bit still needs roughly 5 GB for weights alone. On a 4 GB card that cannot work, so tinyforge can hand the job to a layer-streaming backend:

```yaml
# job.yaml  (this is spec/examples/soup-8b-streaming.yaml in the repo)
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

A 200-step run of this kind trained Llama-3.1-8B in about 9.5 minutes on the 4 GB laptop (it needs about 4 GB of free system RAM and Soup installed in its own environment; `tinyforge memory --params-b 8` shows the plan for your machine). (Caveat: this backend is beta upstream and I have only verified it on Windows.)

Data for that run came from the built-in table tool, which only keeps training examples whose SQL actually executes:

```bash
tinyforge data tabular people.csv --out tab.jsonl --count 500
tinyforge data pii-scan tab.jsonl --redact-to clean.jsonl   # exits 1 if it finds a secret
```

---

## Serving and exporting

```bash
tinyforge serve                      # web UI + API on http://127.0.0.1:8000
```

The API is OpenAI-compatible, so existing client libraries work:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $TINYFORGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

If a prompt contains something that looks like a credential, the server refuses it:

```json
{"error":{"message":"The prompt contains a credential-like string; remove it and retry.",
          "type":"invalid_request_error","code":"secret_in_prompt"}}
```

For fast local inference, export to GGUF and serve through llama.cpp:

```bash
tinyforge export gguf --base ./SmolLM2-360M-Instruct --adapter runs/ft/best --quant q8_0
tinyforge serve --engine llamacpp --gguf out/gguf/base-q8_0.gguf --lora-gguf out/gguf/adapter-f16.gguf
```

Quantization is a real trade-off. On my tests `q8_0` is the safe default; `q4_k_m` lost about 5 points of exact match.

---

## Multi-GPU (new in 0.1.1)

If you have two or more GPUs on one machine, add one flag:

```bash
tinyforge ft train --steps 60 --gpus 2
```

Each GPU trains on its own full batch, and the small LoRA gradients are averaged every step. The same works from a job spec with `resources: {gpus: 2}`. When training finishes, tinyforge compares the weights on every GPU and reports `ddp_max_param_divergence`. On a Kaggle 2×T4 machine it was exactly `0.0`: both GPUs held identical weights. NCCL all-reduce measured 7.3 GB/s between the cards, and resuming from a checkpoint kept them in lockstep.

The speedup is the part I have to be straight about: across four runs, two T4s gave **1.2× to 1.4×** the throughput of one, not 2×. I first suspected one GPU waiting on the other, measured it, balanced the work between them, and the waiting dropped from 0.35 s to 0.02 s per step, but the speedup did not move. So the cause is something else, most likely the shared host (four virtual CPUs). I have added a side-by-side test to settle it, and I will not claim near-linear scaling until it does.

Honest status: this is data-parallel only. It does not shard a model across GPUs (so a model too big for one card still needs the layer-streaming backend), and there is no multi-node training yet.

---

## The part that matters: testing on machines that aren't mine

I developed everything on one Windows laptop. That is a trap: it works on my machine proves almost nothing. So I wrote a test notebook and ran the *published package* on a free Colab T4 (Linux) and a Kaggle 2×T4 box. Here is what broke:

**1. The README quickstart failed.** `tinyforge pipeline` saved its checkpoint in `runs/micro/`, but `tinyforge generate` looked in `runs/default/`. Fixed: it now finds the newest checkpoint and says what to do when there is none.

**2. Colab's preinstalled `torchao` crashed fine-tuning.** `peft` refuses `torchao` older than 0.16, and Colab and Kaggle ship 0.10. tinyforge never uses it, so it now hides a too-old `torchao` from its own process. Users no longer need a workaround.

**3. A broken TensorFlow crashed every command.** Installing the llama.cpp converter downgrades `protobuf`, which breaks the TensorFlow that Colab preinstalls, and `transformers` imports TensorFlow if it can. tinyforge is PyTorch-only, so it now switches TensorFlow, Flax and JAX off for `transformers`.

**4. The T4 was running 2.7× slower than it should.** PyTorch reports `bf16_supported = True` on a T4, but only by *emulating* bf16. tinyforge believed it and trained in emulated bf16. Detecting real support (compute capability 8 or higher) and falling back to fp16 took single-GPU throughput from about 1,100 to about 2,960 tokens/s, and peak memory from 9.6 GB to 2.4 GB.

**5. That fix created a new bug.** In fp16, held-out loss occasionally came out as `NaN` while training stayed finite: a model trained in bf16 can overflow in fp16 on some inputs. Evaluation now recomputes an overflowing batch in bf16 and tells you it did (diagnostic `FT007`).

**6. My own multi-GPU design was slower than it should be.** The first Kaggle run gave only a 1.25× speedup on two GPUs. I had split one fixed batch across the GPUs, so each GPU got half a batch and sat under-utilised. The default is now standard data-parallel (each GPU gets a full batch), with `--split-batch` for the old behaviour.

**7. My own test notebook lied.** It reported a failing test suite as passing because a `| tail` hid the exit code. That is a good reminder that a green checkmark is only as honest as the thing producing it.

None of these showed up on my laptop. All of them would have hit a real user in the first five minutes.

---

## What tinyforge does *not* do (yet)

- **No sharded training.** A model too big for one GPU across several GPUs (FSDP) is not built.
- **No multi-node training**, and Kubernetes with GPUs is untested. The Helm chart and job-spec controller pieces were verified on a CPU-only `kind` cluster only.
- **DPO / preference tuning** is listed but has never been run.
- **Gains depend on the task.** Big wins on narrow, templated tasks; modest on general chat tuning; about nothing on harder question shapes I tried.
- **The speed estimator is uncalibrated.** Treat its tokens-per-second numbers as rough.
- **Windows multi-GPU** is development-only (Windows has no NCCL).

---

## Try it

```bash
pip install tinyforge
tinyforge doctor
tinyforge ft pipeline --steps 100
```

There are ready-made test notebooks for Google Colab and Kaggle in the repo (`notebooks/`), and a step-by-step Linux/Windows install guide in `docs/install.md`. If something breaks on your hardware, that is exactly the information I want: open an issue at https://github.com/rajratangm/tinyforge.

*tinyforge is Apache-2.0 licensed. It is early alpha (0.1.x). The numbers above are single runs on small test sets, and the raw JSON for each is committed in the repo's `benchmarks/` folder.*
