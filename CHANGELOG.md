# Changelog
Format: Keep a Changelog (https://keepachangelog.com), versioning: SemVer. Pre-1.0: minor versions may break.

## [Unreleased]
### Added
- Optional Soup training backend (`spec.backend: soup`): layer streaming trains models larger than VRAM (8B on
  a 4 GB GPU measured); isolated subprocess, telemetry off, allow-listed outputs. See docs/soup-backend.md.
- `tinyforge memory` (VRAM/RAM/disk probe and weight placement plan), chunked cross-entropy, memory-tier cost model.
- Data pipeline: `data ingest` (extract/clean/chunk/dedupe, PDF/Office via the `docs` extra), `data tabular`
  (execution-verified text-to-SQL), `data pairs` (split-before-generation, grounded teacher Q&A), `data pii-scan`
  and `--pii flag|redact|drop`.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` (JSON + SSE) with guardrails v1 (secret block/redact,
  optional PII policy and denylist, counters on /metrics).
- Host RAM/swap gauges on /metrics and a training Grafana dashboard.
- API bearer auth, SQLite job queue, /metrics, TLS/CORS/rate limiting.
- `forgectl` (validate, plan, doctor, run, agent, net check); JobSpec v1alpha1; Helm chart and CRD; Terraform.
- Supply chain: SHA-pinned Actions, govulncheck, gitleaks, Dependabot, hashed dependency locks.

### Fixed
- Fine-tuning now ends replies with the chat template's turn terminator instead of the tokenizer's EOS (wrong stop
  token for base models such as Qwen2.5 base).
- ETL boilerplate filter no longer drops repeated sentences, table rows or code lines; PDF line wraps are rejoined.
