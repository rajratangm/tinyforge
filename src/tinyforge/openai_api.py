"""OpenAI-compatible serving API: GET /v1/models, POST /v1/chat/completions (JSON or SSE streaming).

The router is backend-agnostic: a Backend turns chat messages into a stream of text deltas. `HFBackend`
serves the fine-tuned adapter in runs/ft; tests inject a fake. Auth and GPU-busy checks come from the host.

Guardrail hooks: `input_filters` run on the request messages before generation and may mutate them or raise
`Blocked`; `output_filters` run on the final text. When output filters exist, a streaming request is buffered,
filtered, then replayed as SSE (so streaming cannot bypass a filter, at the cost of time-to-first-token).
Both lists are empty unless the host configures guardrails (see guardrails.py).

Limits: concurrency is the backend's `max_concurrency` (1 for in-process models, so a second request gets
429 + Retry-After at once; a batching engine such as llama.cpp advertises its slot count and a short
`queue_wait_s` during which a request waits for a free slot), `n` must be 1, no tools,
no logprobs. A client disconnect stops the response but the generation thread runs to max_tokens.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engines import UpstreamError

MAX_MESSAGES = 64
MAX_CHARS = 32_000
MAX_NEW_CAP = 2048


class Backend(Protocol):
    name: str

    def stream(self, messages: list[dict], max_tokens: int, temperature: float,
               top_p: float) -> Iterator[str]: ...

    def count_tokens(self, text: str) -> int: ...


class Blocked(Exception):
    """Raised by a guardrail filter to refuse a request or response."""

    def __init__(self, message: str, code: str = "content_filter"):
        super().__init__(message)
        self.code = code


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_tokens: int = Field(256, ge=1, le=MAX_NEW_CAP)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1


def _err(status: int, message: str, type_: str, code: str | None = None,
         headers: dict | None = None):
    return JSONResponse({"error": {"message": message, "type": type_, "code": code}}, status,
                        headers=headers)


def apply_stop(deltas: Iterator[str], stops: list[str]) -> Iterator[tuple[str, bool]]:
    """Yield (text, stopped). Holds back just enough tail to catch a stop string split across deltas."""
    stops = [s for s in stops if s]
    if not stops:
        for d in deltas:
            yield d, False
        return
    hold = max(len(s) for s in stops) - 1
    buf = ""
    for d in deltas:
        buf += d
        cut = min((i for i in (buf.find(s) for s in stops) if i >= 0), default=-1)
        if cut >= 0:
            if buf[:cut]:
                yield buf[:cut], False
            yield "", True
            return
        if len(buf) > hold:
            yield buf[:len(buf) - hold], False
            buf = buf[len(buf) - hold:]
    if buf:
        yield buf, False


def build_router(auth: Callable[..., None], get_backend: Callable[[], Backend],
                 busy: Callable[[], bool] = lambda: False,
                 input_filters: list[Callable[[list[dict]], None]] | None = None,
                 output_filters: list[Callable[[str], str]] | None = None) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth)])
    slots, inflight = threading.Condition(), [0]

    def acquire(backend: Backend) -> bool:
        """Admit a request if a slot is free, waiting up to `queue_wait_s` (0 = reject at once)."""
        limit = getattr(backend, "max_concurrency", 1)
        deadline = time.monotonic() + getattr(backend, "queue_wait_s", 0.0)
        with slots:
            while inflight[0] >= limit:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                slots.wait(left)
            inflight[0] += 1
            return True

    def release() -> None:
        with slots:
            inflight[0] -= 1
            slots.notify()
    in_filters = input_filters if input_filters is not None else []
    out_filters = output_filters if output_filters is not None else []

    @router.get("/models")
    def models():
        b = get_backend()
        return {"object": "list", "data": [{"id": b.name, "object": "model", "created": 0,
                                            "owned_by": "tinyforge"}]}

    @router.post("/chat/completions")
    def chat(req: ChatRequest):
        if req.n != 1:
            return _err(400, "Only n=1 is supported.", "invalid_request_error", "unsupported_n")
        msgs = [m.model_dump() for m in req.messages]
        if sum(len(m["content"]) for m in msgs) > MAX_CHARS:
            return _err(400, f"Messages exceed {MAX_CHARS} characters.", "invalid_request_error", "too_long")
        if msgs[-1]["role"] != "user":
            return _err(400, "The last message must have role 'user'.", "invalid_request_error",
                        "bad_messages")
        try:
            for f in in_filters:
                f(msgs)
        except Blocked as e:
            return _err(400, str(e), "invalid_request_error", e.code)
        if busy():
            return _err(503, "The GPU is busy with a running job.", "server_error", "gpu_busy",
                        {"Retry-After": "30"})
        try:
            backend = get_backend()
        except FileNotFoundError as e:
            return _err(404, str(e), "invalid_request_error", "model_not_found")
        if not acquire(backend):
            return _err(429, "The engine is at its concurrency limit; retry shortly.",
                        "rate_limit_error", "busy",
                        {"Retry-After": "2"})
        stops = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])[:4]
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        prompt_text = "\n".join(m["content"] for m in msgs)

        def deltas() -> Iterator[tuple[str, bool]]:
            return apply_stop(backend.stream(msgs, req.max_tokens, req.temperature, req.top_p), stops)

        def finish_reason(stopped: bool, n_tokens: int) -> str:
            return "stop" if stopped or n_tokens < req.max_tokens else "length"

        def chunk(delta: dict, finish: str | None = None) -> str:
            body: dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                    "model": backend.name,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(body)}\n\n"

        def replay(text: str, reason: str) -> Iterator[str]:
            yield chunk({"role": "assistant", "content": ""})
            for i in range(0, len(text), 24):
                yield chunk({"content": text[i:i + 24]})
            yield chunk({}, reason)
            yield "data: [DONE]\n\n"

        buffered = bool(out_filters)  # output filters need the whole reply, so a stream is buffered first
        if not req.stream or buffered:
            try:
                parts, stopped = [], False
                for text, hit in deltas():
                    parts.append(text)
                    stopped = stopped or hit
                text = "".join(parts)
                for f in out_filters:
                    text = f(text)
            except Blocked as e:
                return _err(400, str(e), "invalid_request_error", e.code)
            except UpstreamError as e:
                return _err(502, str(e), "server_error", "upstream_error")
            except FileNotFoundError as e:  # engines resolve model files lazily, on first use
                return _err(404, str(e), "invalid_request_error", "model_not_found")
            finally:
                release()
            n_out = backend.count_tokens(text)
            if req.stream:  # buffered path: replay the filtered text as SSE
                return StreamingResponse(replay(text, finish_reason(stopped, n_out)),
                                         media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            return {"id": cid, "object": "chat.completion", "created": created, "model": backend.name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": finish_reason(stopped, n_out)}],
                    "usage": {"prompt_tokens": backend.count_tokens(prompt_text), "completion_tokens": n_out,
                              "total_tokens": backend.count_tokens(prompt_text) + n_out}}

        def sse() -> Iterator[str]:
            try:
                yield chunk({"role": "assistant", "content": ""})
                n, stopped = 0, False
                for text, hit in deltas():
                    stopped = stopped or hit
                    if text:
                        n += 1
                        yield chunk({"content": text})
                yield chunk({}, finish_reason(stopped, n))
                yield "data: [DONE]\n\n"
            except UpstreamError as e:
                err = {"error": {"message": str(e), "type": "server_error", "code": "upstream_error"}}
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            except FileNotFoundError as e:
                err = {"error": {"message": str(e), "type": "invalid_request_error",
                                 "code": "model_not_found"}}
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                release()

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router


class HFBackend:
    """Serves the base model plus the LoRA adapter in `run_dir/best` (loaded on first use)."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.name = "tinyforge-ft"
        self._m: tuple | None = None

    def _load(self) -> tuple:
        if self._m is None:
            if not (self.run_dir / "best").exists():
                raise FileNotFoundError("No fine-tuned model yet (runs/ft/best is missing).")
            from peft import PeftModel

            from . import hardware
            from .finetune import FTConfig, load_base, load_tokenizer

            cfg = FTConfig.model_validate_json((self.run_dir / "ft_config.json").read_text())
            hw = hardware.probe()
            base = load_base(cfg.base_model, cfg.quant, hw.bf16_supported, hw.device)
            self._m = (PeftModel.from_pretrained(base, self.run_dir / "best"),
                       load_tokenizer(cfg.base_model), hw.device)
            self.name = f"{cfg.base_model}+adapter"
        return self._m

    def count_tokens(self, text: str) -> int:
        _, tok, _ = self._load()
        return len(tok(text, add_special_tokens=False).input_ids)

    def stream(self, messages: list[dict], max_tokens: int, temperature: float,
               top_p: float) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        model, tok, device = self._load()
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw: dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": tok.pad_token_id,
                              "streamer": streamer, "do_sample": temperature > 0}
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p)
        t = threading.Thread(target=model.generate, kwargs={**ids, **kw}, daemon=True)
        t.start()
        try:
            yield from streamer
        finally:
            t.join(timeout=5)
