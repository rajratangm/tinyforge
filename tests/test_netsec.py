"""Network hardening: TLS/mTLS serve settings, CORS, security headers, body-size limit, rate limit."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import re
import shutil
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typer.testing import CliRunner

from tinyforge import cli, server
from tinyforge.netsec import (
    AuthFailureLimiter,
    NetSettings,
    ServeConfigError,
    _min_tls12,
    build_csp,
    install,
    is_loopback_host,
    parse_cors_origins,
    resolve_serve,
)

TOKEN = "t0ken-" + "x" * 20  # 26 chars: long enough to avoid the short-token warning
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ENV = {"TINYFORGE_API_TOKEN": TOKEN}
UI_FILE = Path(server.UI_DIR) / "index.html"


class Item(BaseModel):
    text: str


def make_app(settings: NetSettings | None = None, ui: bytes | None = b"<script>boot()</script>") -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.get("/api/thing")
    def thing():
        return {"thing": 1}

    @app.post("/echo")
    async def echo(request: Request):
        return {"n": len(await request.body())}

    @app.post("/item")
    def item(i: Item):
        return {"n": len(i.text)}

    install(app, settings or NetSettings(), ui)
    return app


@pytest.fixture
def files(tmp_path):
    out = {n: tmp_path / n for n in ("cert.pem", "key.pem", "ca.pem")}
    for p in out.values():
        p.write_text("placeholder")  # existence is all the validation looks at
    return out


@pytest.fixture(autouse=True)
def clean_limiter():
    server._auth_limiter.reset()
    yield
    server._auth_limiter.reset()


# ---------------------------------------------------------------- settings


def test_cors_origins_parsing():
    assert parse_cors_origins("") == ()
    assert parse_cors_origins(" https://a.example/ , http://localhost:5173,https://a.example ") == (
        "https://a.example", "http://localhost:5173")


@pytest.mark.parametrize("raw", ["*", "https://a.example,*", "https://*.example.com", "a.example",
                                 "ftp://a.example", "https://a.example/path", "https://a.example?x=1"])
def test_cors_origins_rejects_wildcards_and_non_origins(raw):
    with pytest.raises(ValueError):
        parse_cors_origins(raw)


def test_settings_defaults_and_validation():
    s = NetSettings.from_env({})
    assert (s.cors_origins, s.max_body_bytes, s.auth_fail_limit, s.docs_enabled) == ((), 1_048_576, 10, False)
    assert NetSettings.from_env({"TINYFORGE_MAX_BODY_BYTES": "2048", "TINYFORGE_DOCS": "on"}).docs_enabled
    for bad in ({"TINYFORGE_MAX_BODY_BYTES": "lots"}, {"TINYFORGE_MAX_BODY_BYTES": "0"},
                {"TINYFORGE_AUTH_FAIL_LIMIT": "-1"}, {"TINYFORGE_CORS_ORIGINS": "*"}):
        with pytest.raises(ValueError):
            NetSettings.from_env(bad)


# ---------------------------------------------------------------- CSP and headers


def test_csp_hashes_inline_script_and_never_allows_unsafe_script():
    body = b"\nconsole.log('hi');\n"
    csp = build_csp(b"<html><script>" + body + b"</script><script src='x.js'></script></html>")
    digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
    script_src = next(d for d in csp.split("; ") if d.startswith("script-src"))
    assert f"'sha256-{digest}'" in script_src and "unsafe" not in script_src
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert "script-src 'none'" in build_csp(b"<html></html>")


def test_real_ui_script_is_allowed_by_hash():
    html = UI_FILE.read_text(encoding="utf-8")
    bodies = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(bodies) == 1, "the CSP assumes one inline script; update netsec.build_csp if the UI changes"
    digest = base64.b64encode(hashlib.sha256(bodies[0].encode("utf-8")).digest()).decode()
    r = TestClient(server.app).get("/")
    assert r.status_code == 200 and f"'sha256-{digest}'" in r.headers["content-security-policy"]
    handlers = r"\son(click|change|input|submit|load|error)=\""
    assert not re.search(handlers, html), "inline event handlers would need script 'unsafe-inline'"


def test_security_headers_on_every_response_and_hsts_only_over_tls():
    app = make_app()
    http = TestClient(app).get("/ping")
    for name, want in (("x-content-type-options", "nosniff"), ("x-frame-options", "DENY"),
                       ("referrer-policy", "no-referrer")):
        assert http.headers[name] == want
    assert "content-security-policy" in http.headers
    assert "strict-transport-security" not in http.headers
    assert "cache-control" not in http.headers  # only the API is no-store

    assert TestClient(app).get("/api/thing").headers["cache-control"] == "no-store"
    assert TestClient(app).get("/missing").headers["x-frame-options"] == "DENY"  # errors are covered too
    https = TestClient(app, base_url="https://testserver").get("/ping")
    assert https.headers["strict-transport-security"].startswith("max-age=")


def test_real_app_is_hardened_and_docs_are_off_by_default():
    c = TestClient(server.app)
    r = c.get("/healthz")
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff"
    assert c.get("/api/runs").headers["cache-control"] == "no-store"  # even the 401
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


# ---------------------------------------------------------------- CORS


def test_cors_is_same_origin_only_by_default():
    c = TestClient(make_app())
    r = c.get("/ping", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    evil = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    pre = c.options("/echo", headers=evil)
    assert "access-control-allow-origin" not in pre.headers


def test_cors_allow_list_allows_only_listed_origins_without_credentials():
    c = TestClient(make_app(NetSettings(cors_origins=("https://ui.example",))))
    ok = c.get("/ping", headers={"Origin": "https://ui.example"})
    assert ok.headers["access-control-allow-origin"] == "https://ui.example"
    assert "access-control-allow-credentials" not in ok.headers
    pre = c.options("/echo", headers={"Origin": "https://ui.example", "Access-Control-Request-Method": "POST",
                                      "Access-Control-Request-Headers": "authorization"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "https://ui.example"
    other = c.get("/ping", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    evil = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    bad_pre = c.options("/echo", headers=evil)
    assert bad_pre.status_code == 400 and "access-control-allow-origin" not in bad_pre.headers
    assert bad_pre.headers["x-frame-options"] == "DENY"  # headers wrap CORS responses too


# ---------------------------------------------------------------- body size limit


def test_body_over_content_length_limit_is_413():
    c = TestClient(make_app(NetSettings(max_body_bytes=1000)))
    assert c.post("/echo", content=b"x" * 999).json() == {"n": 999}
    r = c.post("/echo", content=b"x" * 1001)
    assert r.status_code == 413 and "1000" in r.json()["detail"]
    assert r.headers["x-frame-options"] == "DENY"


def test_streamed_body_without_content_length_is_413():
    c = TestClient(make_app(NetSettings(max_body_bytes=1000)))
    r = c.post("/echo", content=iter([b"x" * 600, b"y" * 600]))  # chunked: no Content-Length to check
    assert r.status_code == 413
    assert c.post("/echo", content=iter([b"x" * 400, b"y" * 400])).json() == {"n": 800}


def test_declared_body_model_gets_413_not_400():
    c = TestClient(make_app(NetSettings(max_body_bytes=1000)))
    big = ('{"text": "' + "a" * 1200 + '"}').encode()
    assert c.post("/item", content=iter([big[:700], big[700:]]),
                  headers={"content-type": "application/json"}).status_code == 413
    assert c.post("/item", json={"text": "ok"}).json() == {"n": 2}


# ---------------------------------------------------------------- auth failure rate limit


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_limiter_blocks_after_limit_and_recovers_after_window():
    clock = FakeClock()
    lim = AuthFailureLimiter(limit=3, window_s=60, clock=clock)
    for _ in range(3):
        assert lim.blocked("a") is None
        lim.record_failure("a")
        clock.t += 1
    assert lim.blocked("a") == 57 and lim.blocked("b") is None
    clock.t += 57
    assert lim.blocked("a") is None  # the oldest failure aged out


def test_limiter_memory_is_bounded_and_can_be_disabled():
    lim = AuthFailureLimiter(limit=1, window_s=60, max_keys=5)
    for i in range(50):
        lim.record_failure(f"ip{i}")
    assert len(lim._hits) == 5
    off = AuthFailureLimiter(limit=0)
    off.record_failure("a")
    assert off.blocked("a") is None


@pytest.fixture
def api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYFORGE_API_TOKEN", TOKEN)
    monkeypatch.delenv("TINYFORGE_AUTH", raising=False)
    monkeypatch.setattr(server, "ROOT", tmp_path)


def test_wrong_tokens_are_rate_limited_per_client(api_env):
    c = TestClient(server.app)
    for _ in range(10):
        assert c.get("/api/runs", headers={"Authorization": "Bearer guess"}).status_code == 401
    r = c.get("/api/runs", headers={"Authorization": "Bearer guess"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    # blocked means blocked: even the right token must not confirm a guess during the lockout
    assert c.get("/api/runs", headers=AUTH).status_code == 429
    assert c.get("/metrics", headers=AUTH).status_code == 429
    # requests with no token at all are not guesses: still a plain 401, never counted
    assert c.get("/api/runs").status_code == 401
    server._auth_limiter.reset()
    assert c.get("/api/runs", headers=AUTH).status_code == 200


def test_missing_token_and_other_schemes_are_never_counted(api_env):
    c = TestClient(server.app)
    for _ in range(25):
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Basic abc"}).status_code == 401
    assert c.get("/api/runs", headers=AUTH).status_code == 200


# ---------------------------------------------------------------- serve settings


def test_loopback_detection():
    for h in ("127.0.0.1", "127.9.9.9", "localhost", "LOCALHOST", "::1", "[::1]"):
        assert is_loopback_host(h), h
    for h in ("0.0.0.0", "::", "192.168.1.5", "10.0.0.1", "example.com", ""):
        assert not is_loopback_host(h), h


def test_loopback_needs_no_tls_and_no_flags():
    cfg = resolve_serve("127.0.0.1", 8000, env={})
    assert (cfg.tls, cfg.mtls, cfg.uvicorn, cfg.warnings) == (False, False, {}, [])


def test_non_loopback_without_tls_is_refused():
    with pytest.raises(ServeConfigError, match="without TLS"):
        resolve_serve("0.0.0.0", 8000, env=ENV)


def test_insecure_http_is_an_explicit_loud_opt_in():
    cfg = resolve_serve("0.0.0.0", 8000, insecure_http=True, env=ENV)
    assert not cfg.tls and any(w.startswith("INSECURE") for w in cfg.warnings)


def test_non_loopback_always_needs_a_token_and_never_auth_off():
    with pytest.raises(ServeConfigError, match="API token"):
        resolve_serve("0.0.0.0", 8000, insecure_http=True, env={})
    with pytest.raises(ServeConfigError, match="TINYFORGE_AUTH=off"):
        resolve_serve("0.0.0.0", 8000, insecure_http=True, env={**ENV, "TINYFORGE_AUTH": "off"})
    cfg = resolve_serve("0.0.0.0", 8000, insecure_http=True, env={"TINYFORGE_API_TOKEN": "short"})
    assert any("only 5 characters" in w for w in cfg.warnings)


def test_tls_settings_and_minimum_version(files):
    cert, key = files["cert.pem"], files["key.pem"]
    cfg = resolve_serve("0.0.0.0", 8443, ssl_certfile=cert, ssl_keyfile=key, env=ENV)
    assert cfg.tls and not cfg.mtls and not any("INSECURE" in w for w in cfg.warnings)
    assert cfg.uvicorn["ssl_certfile"] == str(cert) and cfg.uvicorn["ssl_keyfile"] == str(key)
    assert "ssl_cert_reqs" not in cfg.uvicorn
    factory = cfg.uvicorn["ssl_context_factory"]
    assert factory is _min_tls12
    ctx = factory(None, lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER))
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2


def test_mtls_settings(files):
    cfg = resolve_serve("0.0.0.0", 8443, ssl_certfile=files["cert.pem"], ssl_keyfile=files["key.pem"],
                        ssl_ca_certs=files["ca.pem"], client_cert_required=True, env=ENV)
    assert cfg.mtls and cfg.uvicorn["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert cfg.uvicorn["ssl_ca_certs"] == str(files["ca.pem"])


def test_inconsistent_tls_flags_are_refused(files, tmp_path):
    with pytest.raises(ServeConfigError, match="--ssl-keyfile needs"):
        resolve_serve("127.0.0.1", 1, ssl_keyfile=files["key.pem"], env={})
    with pytest.raises(ServeConfigError, match="file not found"):
        resolve_serve("127.0.0.1", 1, ssl_certfile=tmp_path / "nope.pem", env={})
    with pytest.raises(ServeConfigError, match="mTLS"):
        resolve_serve("127.0.0.1", 1, client_cert_required=True, env={})
    with pytest.raises(ServeConfigError, match="mTLS"):
        resolve_serve("127.0.0.1", 1, ssl_certfile=files["cert.pem"], client_cert_required=True, env={})
    with pytest.raises(ServeConfigError, match="only applies to mTLS"):
        resolve_serve("127.0.0.1", 1, ssl_certfile=files["cert.pem"], ssl_ca_certs=files["ca.pem"], env={})


def test_bad_net_env_is_a_clean_serve_error():
    with pytest.raises(ServeConfigError, match="wildcards"):
        resolve_serve("127.0.0.1", 8000, env={"TINYFORGE_CORS_ORIGINS": "*"})


# ---------------------------------------------------------------- the `serve` command


def test_serve_command_refuses_unsafe_bind_without_starting(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(k))
    for k in ("TINYFORGE_API_TOKEN", "TINYFORGE_AUTH", "TINYFORGE_SSL_CERTFILE"):
        monkeypatch.delenv(k, raising=False)
    r = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0"])
    assert r.exit_code == 2 and "without TLS" in r.output and calls == []


def test_serve_command_passes_tls_from_env_and_prints_warnings(monkeypatch, files):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    env = {"TINYFORGE_SSL_CERTFILE": str(files["cert.pem"]), "TINYFORGE_SSL_KEYFILE": str(files["key.pem"]),
           "TINYFORGE_API_TOKEN": "short"}
    r = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0", "--port", "9443"], env=env)
    assert r.exit_code == 0, r.output
    (args, kw), = calls
    assert args == ("tinyforge.server:app",) and kw["host"] == "0.0.0.0" and kw["port"] == 9443
    assert kw["ssl_certfile"] == str(files["cert.pem"]) and "only 5 characters" in r.output


def test_serve_command_loopback_default_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(k))
    for k in ("TINYFORGE_SSL_CERTFILE", "TINYFORGE_SSL_KEYFILE", "TINYFORGE_CORS_ORIGINS"):
        monkeypatch.delenv(k, raising=False)
    assert CliRunner().invoke(cli.app, ["serve"]).exit_code == 0
    assert calls == [{"host": "127.0.0.1", "port": 8000}]


# ---------------------------------------------------------------- real TLS / mTLS handshakes


def _find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    for p in (r"C:\Program Files\Git\usr\bin\openssl.exe", r"C:\Program Files\Git\mingw64\bin\openssl.exe"):
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """Throwaway CA + server cert (127.0.0.1/localhost) + client cert, made with the openssl CLI."""
    exe = _find_openssl()
    if not exe:
        pytest.skip("openssl CLI not available: real TLS handshake tests skipped")
    d = tmp_path_factory.mktemp("pki")
    env = {**os.environ}

    def run(*args):
        subprocess.run([exe, *args], cwd=d, env=env, check=True, capture_output=True, timeout=60)

    try:
        run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.crt",
            "-days", "2", "-subj", "/CN=tinyforge-test-ca")
        server_ext = "subjectAltName=IP:127.0.0.1,DNS:localhost\nextendedKeyUsage=serverAuth\n"
        for name, ext in (("server", server_ext), ("client", "extendedKeyUsage=clientAuth\n")):
            (d / f"{name}.ext").write_text(ext)
            run("req", "-newkey", "rsa:2048", "-nodes", "-keyout", f"{name}.key", "-out", f"{name}.csr",
                "-subj", f"/CN=tinyforge-test-{name}")
            run("x509", "-req", "-in", f"{name}.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial",
                "-out", f"{name}.crt", "-days", "2", "-extfile", f"{name}.ext")
    except (subprocess.CalledProcessError, OSError) as e:
        detail = getattr(e, "stderr", b"") or b""
        pytest.skip(f"openssl could not create test certificates: {e} {detail[-200:]!r}")
    return d


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serving(app, **uvicorn_kwargs):
    port = _free_port()
    server_ = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                            **uvicorn_kwargs))
    t = threading.Thread(target=server_.run, daemon=True)
    t.start()
    end = time.time() + 15
    while not server_.started:
        assert t.is_alive() and time.time() < end, "test server did not start"
        time.sleep(0.05)
    try:
        yield port
    finally:
        server_.should_exit = True
        t.join(15)


def _client_ctx(pki: Path, *, client_cert: bool = False) -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=str(pki / "ca.crt"))
    if client_cert:
        ctx.load_cert_chain(str(pki / "client.crt"), str(pki / "client.key"))
    return ctx


def test_real_tls_handshake_and_hsts(pki):
    cfg = resolve_serve("127.0.0.1", 0, ssl_certfile=pki / "server.crt",
                        ssl_keyfile=pki / "server.key", env={})
    with serving(make_app(), **cfg.uvicorn) as port:
        r = httpx.get(f"https://127.0.0.1:{port}/ping", verify=_client_ctx(pki))
        assert r.status_code == 200 and r.headers["strict-transport-security"].startswith("max-age=")
        with socket.create_connection(("127.0.0.1", port)) as raw, \
                _client_ctx(pki).wrap_socket(raw, server_hostname="127.0.0.1") as tls:
            assert tls.version() in ("TLSv1.2", "TLSv1.3")
        with pytest.raises(httpx.TransportError):  # a client not trusting our CA must fail verification
            httpx.get(f"https://127.0.0.1:{port}/ping")


def test_real_mtls_rejects_clients_without_a_valid_certificate(pki):
    cfg = resolve_serve("127.0.0.1", 0, ssl_certfile=pki / "server.crt", ssl_keyfile=pki / "server.key",
                        ssl_ca_certs=pki / "ca.crt", client_cert_required=True, env={})
    with serving(make_app(), **cfg.uvicorn) as port:
        url = f"https://127.0.0.1:{port}/ping"
        with pytest.raises(httpx.TransportError):
            httpx.get(url, verify=_client_ctx(pki))  # no client certificate
        assert httpx.get(url, verify=_client_ctx(pki, client_cert=True)).status_code == 200
