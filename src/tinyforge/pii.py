"""PII and secret detection for training data: regex candidates + validators, no external services.

Detects emails, phone numbers, payment cards (Luhn), US SSNs, IBANs (mod 97), public IPv4 addresses,
and credential formats (AWS, GitHub, OpenAI-style, Slack, Google API keys, JWTs, private-key blocks).

Policy used by the data pipeline: secrets are always dropped (never train on a credential); other PII is
flagged, redacted to [KIND] placeholders, or the whole example is dropped.

Honest limits: this is pattern matching. It does NOT detect names, street addresses, dates of birth,
medical or free-text identifiers, and non-US national IDs. It will miss obfuscated values
("john at example dot com") and can flag look-alike numbers. For names and addresses use an NER
tool such as Presidio on top; treat a clean scan as "no pattern hits", never as "no personal data".
"""
from __future__ import annotations

import ipaddress
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

SECRET_PATTERNS: dict[str, str] = {
    "aws_key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    "openai_key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "google_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
SECRET_RE = re.compile("|".join(SECRET_PATTERNS.values()))
_SECRET_COMPILED = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}(?!\w)(?!\.\d)")
CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
SSN_RE = re.compile(r"(?<![\d-])(\d{3})-(\d{2})-(\d{4})(?![\d-])")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
DATETIME_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?")
DATE_LIKE =re.compile(r"^\d{1,4}[-./]\d{1,2}[-./]\d{1,4}$")

PII_KINDS = ("email", "phone", "card", "ssn", "iban", "ip")


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    secret: bool = False


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def _phone_ok(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    if not 10 <= len(digits) <= 15 or DATE_LIKE.match(s):
        return False
    return s.startswith("+") or "(" in s or bool(re.search(r"[ .-]", s))


def _public_ip(s: str) -> bool:
    try:
        ip = ipaddress.IPv4Address(s)
    except ValueError:
        return False
    return ip.is_global and not ip.is_multicast


def scan(text: str, kinds: Iterable[str] | None = None) -> list[Finding]:
    """Non-overlapping findings, secrets first, then longer. `kinds` limits PII kinds, never secrets."""
    want = set(PII_KINDS if kinds is None else kinds)
    found: list[Finding] = []
    for kind, rx in _SECRET_COMPILED.items():
        found += [Finding(kind, m.start(), m.end(), True) for m in rx.finditer(text)]
    if "email" in want:
        found += [Finding("email", m.start(), m.end()) for m in EMAIL_RE.finditer(text)]
    if "card" in want:
        for m in CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn(digits) and len(set(digits)) > 1:
                found.append(Finding("card", m.start(), m.end()))
    if "ssn" in want:
        for m in SSN_RE.finditer(text):
            area, grp, serial = m.groups()
            if area not in ("000", "666") and not area.startswith("9") and grp != "00" and serial != "0000":
                found.append(Finding("ssn", m.start(), m.end()))
    if "iban" in want:
        found += [Finding("iban", m.start(), m.end()) for m in IBAN_RE.finditer(text) if _iban_ok(m.group())]
    if "ip" in want:
        found += [Finding("ip", m.start(), m.end()) for m in IPV4_RE.finditer(text) if _public_ip(m.group())]
    if "phone" in want:
        # card-shaped runs and dates/timestamps are not phone numbers
        not_phone = [m.span() for rx in (CARD_RE, DATETIME_RE) for m in rx.finditer(text)]
        found += [Finding("phone", m.start(), m.end()) for m in PHONE_RE.finditer(text)
                  if _phone_ok(m.group().strip())
                  and not any(m.start() < e and s < m.end() for s, e in not_phone)]
    found.sort(key=lambda f: (not f.secret, -(f.end - f.start), f.start))
    kept: list[Finding] = []
    for f in found:
        if all(f.end <= k.start or f.start >= k.end for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def redact(text: str, findings: list[Finding] | None = None) -> str:
    out, pos = [], 0
    for f in findings if findings is not None else scan(text):
        out.append(text[pos:f.start])
        out.append(f"[{f.kind.upper()}]")
        pos = f.end
    out.append(text[pos:])
    return "".join(out)


def mask(value: str) -> str:
    """Safe-to-print form of a match: keeps 2 chars at each end only for longer values."""
    return value[:2] + "*" * (len(value) - 4) + value[-2:] if len(value) > 8 else "*" * len(value)


def summarize(texts: Iterable[str]) -> tuple[Counter, int, dict[str, list[str]]]:
    """(counts by kind, number of texts with any finding, up to 3 masked examples per kind)."""
    counts: Counter = Counter()
    examples: dict[str, list[str]] = {}
    hit = 0
    for t in texts:
        fs = scan(t)
        if fs:
            hit += 1
        for f in fs:
            counts[f.kind] += 1
            ex = examples.setdefault(f.kind, [])
            if len(ex) < 3:
                ex.append(mask(t[f.start:f.end]))
    return counts, hit, examples
