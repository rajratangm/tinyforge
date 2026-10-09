# How tinyforge compares

Short version: tinyforge is **not** a faster trainer than the established tools, and it does not try to be. Use it for
what it adds around training, on small hardware. Use the others when you need their strengths.

The other projects move quickly. Check their own docs before deciding; this page only states what we are
sure of about tinyforge and gives pointers, not scorecards, for the rest.

## Where the others are stronger

- **[Unsloth](https://github.com/unslothai/unsloth)** is built around speed and memory savings with custom kernels. If raw training speed on one GPU is your priority, benchmark it first.
- **[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)** has broad training-method coverage, YAML configs and distributed training (FSDP, DeepSpeed). tinyforge has data-parallel LoRA on one machine only and **no FSDP or multi-node** yet (see issue #16).
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** has a wide model list and a browser UI for launching training.

## What tinyforge is for

| | tinyforge |
|---|---|
| Quality gates | the eval **fails** (non-zero exit) when the tuned model is not better than base, when general-text loss shows forgetting, when the base is untrained (`FE010`) or output loops (`FE011`) |
| Small GPUs | the planner picks 4-bit vs fp16, checkpointing and a token budget to fit VRAM and recovers from OOM; an 8B model trains on a 4 GB card through the optional `soup` layer-streaming backend |
| Plain-English warnings | every stage returns coded warnings with a fix (`HW003`, `FD007`, `FT005` ...) |
| After training | GGUF export, llama.cpp or an OpenAI-compatible server with guardrails, a dashboard with latency and KV-cache charts |
| Data | ingest, de-duplicate, PII/credential scan, and text-to-SQL examples verified by executing the SQL |
| Windows | developed and tested first on Windows; also tested on a Colab T4 and a Kaggle 2x T4 |

## Where tinyforge is weaker, today

- Young alpha with one maintainer and a small test footprint: single GPU for the headline results.
- Multi-GPU scales only **1.2–1.4x** on 2x T4 (issue #15).
- Our fine-tuning results use templated text-to-SQL tests: they show the pipeline works, not general model quality. See [Verified results](../README.md#verified-results-what-was-actually-run).
- Methods are centred on LoRA/QLoRA supervised fine-tuning; `methods suggest` can recommend other methods (DPO, ORPO) but the native trainer does not run them. No vision models, no multi-node.

Missing something here or wrong about another tool? Open an issue or PR.
