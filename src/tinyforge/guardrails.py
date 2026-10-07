"""Serving guardrails v1: pattern-based input/output filters for the OpenAI-compatible endpoint.

Configured from the environment (read once at startup; a bad value fails fast):
  TINYFORGE_GUARD_SECRETS      block (default) | off
                               credential-like strings: in a prompt -> 400, in a reply -> redacted
  TINYFORGE_GUARD_PII_INPUT    off (default) | redact | block
  TINYFORGE_GUARD_PII_OUTPUT   off (default) | redact
  TINYFORGE_GUARD_DENYLIST     path to a file of case-insensitive regexes, one per line (# comments)
                               a match in a prompt -> 400 content_filter, in a reply -> 400 output_denied

What this is NOT: no prompt-injection detection, no jailbreak or toxicity classifier, no topical policy, no
semantic checks. PII detection is the pattern scanner in pii.py (no names/addresses). Treat this as a seatbelt
against accidental credential leaks and obvious policy phrases, not as a safety system; put a classifier such
as Llama Guard behind the same hooks for that.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import pii
from .openai_api import Blocked

COUNTS: Counter = Counter()  # process-wide; exported on /metrics as tinyforge_guard_events_total{reason}


@dataclass
class GuardConfig:
    secrets: str = "block"          # block | off
    pii_input: str = "off"          # off | redact | block
    pii_output: str = "off"         # off | redact
    deny: list[re.Pattern] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict | None = None) -> GuardConfig:
        e = os.environ if env is None else env

        def choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
            v = e.get(name, default).strip().lower()
            if v not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {v!r}")
            return v

        deny: list[re.Pattern] = []
        path = e.get("TINYFORGE_GUARD_DENYLIST", "").strip()
        if path:
            for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        deny.append(re.compile(line, re.I))
                    except re.error as err:
                        raise ValueError(f"{path}:{n}: invalid regex ({err})") from err
        return cls(choice("TINYFORGE_GUARD_SECRETS", "block", ("block", "off")),
                   choice("TINYFORGE_GUARD_PII_INPUT", "off", ("off", "redact", "block")),
                   choice("TINYFORGE_GUARD_PII_OUTPUT", "off", ("off", "redact")), deny)

    @property
    def active(self) -> bool:
        return bool(self.secrets == "block" or self.pii_input != "off" or self.pii_output != "off"
                    or self.deny)


def build_filters(cfg: GuardConfig, counts: Counter = COUNTS) -> tuple[list[Callable], list[Callable]]:
    def input_filter(msgs: list[dict]) -> None:
        for m in msgs:
            text = m["content"]
            if any(rx.search(text) for rx in cfg.deny):
                counts["denied_input"] += 1
                raise Blocked("The request matches a blocked pattern.", "content_filter")
            found = pii.scan(text, kinds=pii.PII_KINDS if cfg.pii_input != "off" else ())
            if cfg.secrets == "block" and any(f.secret for f in found):
                counts["secret_in_prompt"] += 1
                raise Blocked("The prompt contains a credential-like string; remove it and retry.",
                              "secret_in_prompt")
            pii_found = [f for f in found if not f.secret]
            if pii_found and cfg.pii_input == "block":
                counts["pii_in_prompt"] += 1
                kinds = ", ".join(sorted({f.kind for f in pii_found}))
                raise Blocked(f"The prompt contains personal data ({kinds}); remove it and retry.",
                              "pii_in_prompt")
            if pii_found and cfg.pii_input == "redact":
                counts["pii_redacted_input"] += 1
                m["content"] = pii.redact(text, found if cfg.secrets == "block" else pii_found)

    def output_filter(text: str) -> str:
        if any(rx.search(text) for rx in cfg.deny):
            counts["denied_output"] += 1
            raise Blocked("The response was withheld by a content filter.", "output_denied")
        kinds = pii.PII_KINDS if cfg.pii_output != "off" else ()
        found = pii.scan(text, kinds=kinds)
        if cfg.secrets != "block":
            found = [f for f in found if not f.secret]
        if not found:
            return text
        for f in found:
            counts["secret_redacted_output" if f.secret else "pii_redacted_output"] += 1
        return pii.redact(text, found)

    ins = [input_filter] if (cfg.secrets == "block" or cfg.pii_input != "off" or cfg.deny) else []
    outs = [output_filter] if (cfg.secrets == "block" or cfg.pii_output != "off" or cfg.deny) else []
    return ins, outs
