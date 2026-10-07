# Inference techniques (llama.cpp engine)

`tinyforge serve --engine llamacpp` exposes llama.cpp's inference options through environment variables (or the
`LlamaCppBackend` arguments). Measured on a 3B Q4_K_M model, RTX 3050 Ti 4 GB, all layers on the GPU, greedy decoding
(`benchmarks/inference-techniques-qwen3b.json`):

| Option | Setting | Effect measured | Use it when |
|---|---|---|---|
| Flash attention | `TINYFORGE_FLASH_ATTN=auto\|on\|off` | `off` was ~8% slower (50.3 vs 54.5 tok/s) | leave on `auto` |
| Quantised KV cache | `TINYFORGE_KV_TYPE=f16\|q8_0\|q4_0` | q8_0 halves the KV buffer (144 -> 76.5 MB), ~3.5% slower, output text changed | long contexts or many slots, where the KV cache is the memory problem |
| N-gram speculation | `TINYFORGE_SPECULATIVE=ngram` | lossless; 129 vs 54.5 tok/s on repeated requests, **no gain on the first (cold) request** | repeated or templated outputs (extraction, SQL patterns, RAG answers that copy the context) |
| Draft-model speculation | `TINYFORGE_SPECULATIVE=draft` + `TINYFORGE_DRAFT_GGUF=small.gguf` | not measured here | a small same-family model is available; wiring is tested, speed is not |
| Slots | `TINYFORGE_PARALLEL=N`, `TINYFORGE_QUEUE_WAIT_S` | N requests decode together; extra requests wait, then 429 | several users at once |
| GPU layers | `TINYFORGE_NGL=auto\|N` | `auto` picks from free VRAM | models larger than VRAM |

Honest limits: one model, one prompt, three timed runs per setting; the n-gram speedup is inflated by repeating an
identical request; changing the KV type or attention kernel can change greedy output text (not scored for
correctness). Other llama.cpp options (grammar/JSON-schema constrained decoding, prompt-cache tuning) are not wired
through the API yet.

## Dashboard: memory and KV cache

The "Serving and memory" panel plots, from `/api/telemetry`: GPU/RAM/engine memory over time; the KV cache size
against tokens in context (an exact straight line, bytes/token = 2 x layers x KV heads x head dim x bytes per value,
read from the GGUF header) with your requests as dots; GPU memory per request; and tokens/s per request.
llama.cpp reserves the whole KV buffer at start-up, so measured GPU memory stays flat while the used share grows;
the in-process Hugging Face backend grows its cache with the context. GPU numbers are whole-device.
