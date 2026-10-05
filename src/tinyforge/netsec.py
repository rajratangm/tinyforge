"""Network-facing hardening for the API: TLS/mTLS serve settings, CORS policy, security headers, request-size
limit and an auth-failure rate limiter.

Production guidance: terminate TLS at an ingress / load balancer (cert rotation, WAF, DDoS protection live
there). Use the built-in TLS for single-node installs and mTLS for east-west traffic between components.

The middlewares are plain ASGI (not BaseHTTPMiddleware) so they never buffer streaming responses such as the
job-event SSE stream.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import ipaddress
import math
import os
import re
import ssl
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from starlette.datastructures import MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

DEFAULT_MAX_BODY_BYTES = 1_048_576  # API bodies are small JSON; the largest legitimate one is a few KB
DEFAULT_AUTH_FAIL_LIMIT = 10
DEFAULT_AUTH_FAIL_WINDOW_S = 60.0
MIN_TOKEN_LEN = 16


# ---------------------------------------------------------------- settings from the environment


def parse_cors_origins(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated allow-list. Same-origin only is the default (empty). '*' is never accepted."""
    origins: list[str] = []
    for item in raw.split(","):
        o = item.strip()
        if not o:
            continue
        if "*" in o:
            raise ValueError("TINYFORGE_CORS_ORIGINS must list explicit origins such as "
                             "https://app.example.com; wildcards ('*') are not allowed.")
        parts = urlsplit(o)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/") \
                or parts.query or parts.fragment:
            raise ValueError(f"TINYFORGE_CORS_ORIGINS entry {o!r} is not an origin: use scheme://host[:port] "
                             "with no path.")
        origins.append(f"{parts.scheme}://{parts.netloc}")
    return tuple(dict.fromkeys(origins))


def _int_env(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class NetSettings:
    cors_origins: tuple[str, ...] = ()
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    auth_fail_limit: int = DEFAULT_AUTH_FAIL_LIMIT  # 0 disables the limiter
    auth_fail_window_s: float = DEFAULT_AUTH_FAIL_WINDOW_S
    docs_enabled: bool = False  # /docs, /redoc and /openapi.json are open pages: off unless asked for

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> NetSettings:
        env = os.environ if env is None else env
        return cls(
            cors_origins=parse_cors_origins(env.get("TINYFORGE_CORS_ORIGINS", "")),
            max_body_bytes=_int_env(env, "TINYFORGE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES, 1),
            auth_fail_limit=_int_env(env, "TINYFORGE_AUTH_FAIL_LIMIT", DEFAULT_AUTH_FAIL_LIMIT, 0),
            auth_fail_window_s=float(_int_env(env, "TINYFORGE_AUTH_FAIL_WINDOW_S",
                                              int(DEFAULT_AUTH_FAIL_WINDOW_S), 1)),
            docs_enabled=env.get("TINYFORGE_DOCS", "").lower() in ("1", "on", "true", "yes"),
        )


# ---------------------------------------------------------------- Content-Security-Policy


_INLINE_SCRIPT = re.compile(rb"<script(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script>", re.S | re.I)


def build_csp(ui_html: bytes | None) -> str:
    """Restrictive CSP for the single-file UI.

    Scripts: only the UI's own inline <script>, allowed by SHA-256 hash (no 'unsafe-inline', no eval).
    Styles: 'unsafe-inline' is unavoidable today because the UI uses inline style="" attributes (some built
    through innerHTML), which cannot be hash-allowed. Moving them to classes would let this be tightened.
    """
    hashes = []
    for m in _INLINE_SCRIPT.finditer(ui_html or b""):
        body = m.group(1)
        if body.strip():
            hashes.append("'sha256-" + base64.b64encode(hashlib.sha256(body).digest()).decode() + "'")
    return "; ".join([
        "default-src 'none'",
        "script-src " + (" ".join(hashes) or "'none'"),
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


# ---------------------------------------------------------------- ASGI middlewares


class SecurityHeadersMiddleware:
    """Adds hardening headers to every HTTP response (including CORS preflights and errors)."""

    def __init__(self, app, csp: str) -> None:
        self.app = app
        self.csp = csp

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        tls = scope.get("scheme") == "https"

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h["X-Content-Type-Options"] = "nosniff"
                h["X-Frame-Options"] = "DENY"
                h["Referrer-Policy"] = "no-referrer"
                h["Content-Security-Policy"] = self.csp
                h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
                if path.startswith("/api/") or path == "/metrics":
                    h["Cache-Control"] = "no-store"
                if tls:  # HSTS over plain HTTP is ignored by browsers and misleading; only send it over TLS
                    h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """413 for request bodies over `max_bytes` (a Content-Length check, then a streamed-chunk count)."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    too_big = int(value) > self.max_bytes
                except ValueError:
                    too_big = False  # malformed length: the server's HTTP parser rejects it
                if too_big:
                    resp = JSONResponse({"detail": f"Request body exceeds {self.max_bytes} bytes."},
                                        status_code=413)
                    await resp(scope, receive, send)
                    return
        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI's body parsing re-raises an HTTPException but turns any other exception into a
                    # 400; the exception middleware then renders this as a 413. fastapi's class subclasses
                    # starlette's, so it passes whichever of the two a given FastAPI version checks for.
                    raise HTTPException(413, f"Request body exceeds {self.max_bytes} bytes.")
            return message

        await self.app(scope, counting_receive, send)


def install(app: FastAPI, settings: NetSettings, ui_html: bytes | None = None) -> None:
    """Add CORS (only when an allow-list is configured), the body limit and the security headers.

    add_middleware makes the last one added the outermost, so headers wrap everything, then the size limit,
    then CORS next to the app.
    """
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=False,  # auth is a bearer header, never a cookie
            max_age=600,
        )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware, csp=build_csp(ui_html))


# ---------------------------------------------------------------- auth-failure rate limit


class AuthFailureLimiter:
    """Sliding-window limiter on *wrong bearer tokens* per client key, to blunt token guessing.

    Requests with no token at all are not counted (that is how the UI discovers it needs one). Once a key is
    over the limit every request from it gets 429 until the window passes, even with the right token: a
    limiter that still accepted the right token would confirm a guess. Memory is bounded by `max_keys`.
    """

    def __init__(self, limit: int = DEFAULT_AUTH_FAIL_LIMIT, window_s: float = DEFAULT_AUTH_FAIL_WINDOW_S,
                 max_keys: int = 10_000, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window_s, self.max_keys, self._clock = limit, window_s, max_keys, clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float] | None:
        q = self._hits.get(key)
        if q is None:
            return None
        while q and now - q[0] >= self.window_s:
            q.popleft()
        if not q:
            del self._hits[key]
            return None
        return q

    def blocked(self, key: str) -> int | None:
        """Seconds until the key may try again, or None if it is not blocked."""
        if self.limit <= 0:
            return None
        with self._lock:
            now = self._clock()
            q = self._prune(key, now)
            if q is not None and len(q) >= self.limit:
                return max(1, math.ceil(self.window_s - (now - q[0])))
            return None

    def record_failure(self, key: str) -> None:
        if self.limit <= 0:
            return
        with self._lock:
            now = self._clock()
            q = self._prune(key, now)
            if q is None:
                q = self._hits[key] = deque()
            q.append(now)
            self._hits.move_to_end(key)
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# ---------------------------------------------------------------- `tinyforge serve` settings


class ServeConfigError(ValueError):
    """The requested serve configuration is unsafe or inconsistent; the message says how to fix it."""


def is_loopback_host(host: str) -> bool:
    h = host.strip().strip("[]")
    if h.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False  # any other name could resolve to a public interface: treat as non-loopback


@dataclass
class ServeSettings:
    host: str
    port: int
    uvicorn: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tls: bool = False
    mtls: bool = False


def _min_tls12(_config, default_factory: Callable[[], ssl.SSLContext]) -> ssl.SSLContext:
    ctx = default_factory()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # TLS 1.3 is used whenever the client supports it
    return ctx


def resolve_serve(host: str, port: int, *, ssl_certfile: Path | str | None = None,
                  ssl_keyfile: Path | str | None = None, ssl_ca_certs: Path | str | None = None,
                  client_cert_required: bool = False, insecure_http: bool = False,
                  env: Mapping[str, str] | None = None) -> ServeSettings:
    """Validate the serve flags; return the uvicorn kwargs. Raises ServeConfigError if the setup is unsafe."""
    env = os.environ if env is None else env
    try:
        NetSettings.from_env(env)  # fail with a clear message here instead of a traceback at import time
    except ValueError as e:
        raise ServeConfigError(str(e)) from e

    cfg = ServeSettings(host=host, port=port)
    loopback = is_loopback_host(host)
    tls = bool(ssl_certfile)
    if ssl_keyfile and not ssl_certfile:
        raise ServeConfigError("--ssl-keyfile needs --ssl-certfile.")
    for label, p in (("--ssl-certfile", ssl_certfile), ("--ssl-keyfile", ssl_keyfile),
                     ("--ssl-ca-certs", ssl_ca_certs)):
        if p and not Path(p).is_file():
            raise ServeConfigError(f"{label}: file not found: {p}")
    if client_cert_required and not (tls and ssl_ca_certs):
        raise ServeConfigError("--client-cert-required (mTLS) needs TLS (--ssl-certfile/--ssl-keyfile) and "
                               "--ssl-ca-certs to verify client certificates against.")
    if ssl_ca_certs and not client_cert_required:
        raise ServeConfigError("--ssl-ca-certs only applies to mTLS: add --client-cert-required, or drop it.")

    if not loopback:
        if not tls and not insecure_http:
            raise ServeConfigError(
                f"refusing to listen on {host} without TLS: tokens and data would cross the network in "
                "clear text. Pass --ssl-certfile/--ssl-keyfile, terminate TLS at a reverse proxy and bind to "
                "127.0.0.1, or pass --insecure-http to accept the risk (trusted private network only).")
        if env.get("TINYFORGE_AUTH", "").lower() == "off":
            raise ServeConfigError(f"refusing to listen on {host} with TINYFORGE_AUTH=off: that switch is "
                                   "for local development on loopback only.")
        token = env.get("TINYFORGE_API_TOKEN", "")
        if not token:
            raise ServeConfigError(f"refusing to listen on {host} without an API token: "
                                   "set TINYFORGE_API_TOKEN.")
        if len(token) < MIN_TOKEN_LEN:
            cfg.warnings.append(f"TINYFORGE_API_TOKEN is only {len(token)} characters; use at least "
                                f"{MIN_TOKEN_LEN} random characters on a network-facing server.")
        if insecure_http and not tls:
            cfg.warnings.append(f"INSECURE: serving plain HTTP on {host}. Tokens and data are sent in clear "
                                "text and can be read or modified by anyone on the network path.")

    if tls:
        cfg.tls = True
        cfg.uvicorn["ssl_certfile"] = str(ssl_certfile)
        if ssl_keyfile:
            cfg.uvicorn["ssl_keyfile"] = str(ssl_keyfile)
        if "ssl_context_factory" in inspect.signature(_uvicorn_config()).parameters:
            cfg.uvicorn["ssl_context_factory"] = _min_tls12
        else:
            cfg.warnings.append("this uvicorn version cannot pin a TLS minimum version; relying on the "
                                "Python/OpenSSL default (TLS 1.2+ on Python 3.10+). Upgrade uvicorn to "
                                "enforce it.")
        if client_cert_required:
            cfg.mtls = True
            cfg.uvicorn["ssl_ca_certs"] = str(ssl_ca_certs)
            cfg.uvicorn["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    return cfg


def _uvicorn_config():
    import uvicorn

    return uvicorn.Config
