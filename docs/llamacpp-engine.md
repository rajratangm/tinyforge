# llama.cpp serving engine and GGUF export

## Export: `tinyforge export gguf`

    tinyforge export gguf --base <local HF folder> --adapter runs/x/best --out out/gguf --quant q4_k_m

Writes `adapter-f16.gguf` and `base-<quant>.gguf` (the f16 intermediate is deleted unless `--keep-f16`). Needs the
llama.cpp tools: `TINYFORGE_LLAMACPP_SRC` (source tree with `convert_*_to_gguf.py` and `gguf-py`) and
`TINYFORGE_LLAMACPP_BIN` (folder with `llama-quantize`); both default to `tools/llama.cpp`.

Measured here: Qwen2.5-3B + adapter -> Q4_K_M in 1m41s; Llama-3.1-8B converted by the same steps by hand.

## Serve: `tinyforge serve --engine llamacpp`

    tinyforge serve --engine llamacpp --gguf out/gguf/base-q4_k_m.gguf --lora-gguf out/gguf/adapter-f16.gguf

(or `TINYFORGE_ENGINE=llamacpp TINYFORGE_GGUF=... TINYFORGE_LORA_GGUF=...`). The API is the same `/v1/chat/completions`
with the same auth, guardrails and metrics; the engine only replaces the in-process model.

- `llama-server` is started lazily on the first request, bound to `127.0.0.1`, protected by a random per-launch
  API key, and stopped when tinyforge exits. Find it with `TINYFORGE_LLAMA_SERVER` (else PATH, else `tools/llama.cpp`).
- GPU layers: `TINYFORGE_NGL=auto` (default) picks from free VRAM; a number overrides it.
- Concurrency: `TINYFORGE_PARALLEL` slots (default 2). Extra requests wait up to `TINYFORGE_QUEUE_WAIT_S` seconds
  (default 20) for a slot, then get 429 + `Retry-After`.
- Engine failure -> 502 `upstream_error`; missing GGUF -> 404 `model_not_found`.

## Limits

- The adapter is trained against an NF4/fp16 base and served on a quantized GGUF base. Checked once (8B, 100
  questions, no loss), not in general.
- The engine does not coordinate the GPU with training jobs; stop serving before training on a small card.
- vLLM is not integrated (decision: later). `ProxyBackend` in `engines.py` already speaks the OpenAI protocol, so a
  vLLM backend would be a small subclass plus its own process management on Linux.
- Verified on Windows only.
