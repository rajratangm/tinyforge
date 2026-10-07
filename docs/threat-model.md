# Threat model (STRIDE, per component)

Status: first pass, 2026-10-07, written from the code as it stands. It is a review aid, not a certification.
Mapped lists to use next: OWASP ASVS (API), OWASP Top 10 for LLM Apps, MITRE ATLAS. Not yet done: those mappings,
an external review, and any penetration test.

## Assets
Model weights and adapters, training data (may be private), the API token, node-agent certificates, GPU time.

## 1. API server (`src/tinyforge/server.py`, `netsec.py`)
| | Threat | Mitigation in code | Gap |
|---|---|---|---|
| S | Unauthenticated job submission | Bearer token required on `/api/*`; refuses to serve with no token unless `TINYFORGE_AUTH=off` | Single shared token, no per-user identity or rotation |
| S/D | Token brute force | 429 after repeated failures per client | Per-process memory; resets on restart; behind a proxy all clients may look identical |
| T | Browser-origin abuse | CORS off unless origins listed; CSP and nosniff headers | No CSRF concern while auth is a header, revisit if cookies are added |
| I | Info leak via docs/errors | `/docs`, `/redoc`, `/openapi.json` off unless `TINYFORGE_DOCS=on` | Job logs may contain dataset snippets; not redacted |
| D | Queue flooding | Single running job, FIFO queue in SQLite | No per-client quota or queue length cap |
| E | Arbitrary code via job spec | Spec is validated data, not code | Worker runs with the server's privileges; no sandbox |

## 2. Checkpoints and models
Loading uses `torch.load(weights_only=True)`, which blocks pickle code execution in checkpoints. Not covered: signing or
hash verification of base models pulled from the Hub (use pinned revisions), safetensors-only enforcement, SDC in checkpoints.

## 3. Node agent (`internal/agent`)
TLS and optional mTLS; executes jobs through the worker. Gaps: orphaned worker after a hard kill, no seccomp/AppArmor
profile, no resource limits beyond what the container runtime gives, worker logs not yet exposed.

## 4. Container and cluster (`Dockerfile`, `deploy/`, `infra/terraform`)
Digest-pinned bases, multi-stage, NetworkPolicies verified on kind. Gaps: no image signing or SBOM yet (Q1 next), policies need an
enforcing CNI, Terraform never applied, no admission policy for signed images.

## 5. Supply chain
SHA-pinned Actions, hashed locks, govulncheck, gitleaks, pip-audit (blocking, torch excluded). Open: torch 2.6.0 advisories.
