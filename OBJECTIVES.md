# OBJECTIVES (agent-internal north star; re-read at the start of each step and after each milestone)

Goal: tinyforge becomes the default tool industry reaches for to fine-tune/evaluate/serve small LLMs on modest or
shared hardware. Adoption bar = "at least as good as Axolotl / Unsloth / LLaMA-Factory on something that matters,
and safer/more trustworthy on everything else". Differentiator to protect: planner + warnings + honest evals.

Rules of engagement (learned this project)
- One measurable step at a time; ROADMAP.md holds the user-facing table, this file holds the long tail.
- Never report a gain without a fair baseline (instructed base, not raw base). Report caveats in the same breath.
- Measure before optimizing (VRAM estimator was 5x off; throughput was launch-bound). Calibrate heuristics on hardware.
- Every failure mode gets a stable diagnostic code + fix text. Silent degradation is the enemy (spill, NaN, leakage).
- Verify downloads (sha256), pin versions, no secrets/PII in data or logs.
- Windows quirks: PS blocks `del`/Remove-Item alias patterns in heredocs; write scripts with Write tool; kill only
  .venv processes; HF_HUB_DISABLE_PROGRESS_BARS=1; Start-Process for long jobs + bash until-loop to wait.
- User wants one-line status answers when asked; keep prose short; confirm before big/outward actions.

## A. Install and onboarding (P0)
- `pipx install tinyforge` / `uv tool install`; prebuilt wheels; auto-select torch CUDA wheel; `tinyforge doctor --fix`.
- Docker image on a registry (GHCR/ECR), signed (cosign), SBOM, Trivy scan in CI, slim + GPU variants.
- First-run wizard in UI/CLI: detect GPU -> recommend model + dataset size + expected time/cost before running.
- Offline/air-gapped mode (local model + dataset paths only, no HF calls); proxy + mirror support.

## B. Models and data (P0)
- Support Llama 3.x, Qwen 2.5/3, Mistral, Gemma, Phi, SmolLM via architecture-agnostic path; model compat matrix + CI smoke.
- Gated-model auth (HF token) handled safely; license surfaced in plan (commercial use warning).
- Dataset formats: alpaca, sharegpt, chatml, OpenAI messages, prompt/completion, parquet, CSV, streaming HF; column mapper UI.
- Data quality: near-dup (MinHash), language ID, toxicity/PII scrub (presidio-style), contamination check vs eval sets,
  token-length histogram, label-leak detection, train/val leakage report, license/provenance manifest.
- Synthetic data helper (distill from a bigger API model) with cost cap + provenance tags.
- Packing / sequence packing with correct attention masking; multi-turn masking options; DPO/ORPO/KTO preference data.

## C. Training methods and speed (P1)
- LoRA/QLoRA/DoRA/rsLoRA; full fine-tune for small models; DPO/ORPO/SimPO; continued pretraining; distillation.
- Fused kernels (Triton): SwiGLU, RMSNorm fwd+bwd, cross-entropy chunked (kills 49k-vocab logit memory), rotary, LoRA
  fused matmul. Flash-attn 2/3 where supported. torch.compile where stable. Liger-style kernels as reference.
- Chunked/linear cross-entropy to cut logits memory (biggest 4GB win); paged optimizers; 8-bit Adam; CPU offload.
- Auto hyperparameter sweep on small budget (LR, rank) with early stopping; LR finder.
- Resume everywhere incl. data order; deterministic seeds; spot-interrupt safe checkpoints (S3 sync exists).
- Multi-GPU: DDP, FSDP2, DeepSpeed ZeRO; planner extends to N GPUs; cloud-burst job submit (SageMaker/Batch/EC2 spot).
- JAX backend (same train() interface; WSL2/Linux); MLX backend for Apple silicon; ROCm/Intel support later.

## D. Evaluation and trust (P0, the moat)
- Task suites as plugins: SQL (add execution accuracy), JSON/schema extraction, classification, summarization (judge),
  code (unit tests), RAG faithfulness, safety/refusal, instruction following (IFEval), MMLU/ARC subset for forgetting.
- Always: tuned vs base vs instructed-base vs previous-run; confidence intervals (bootstrap) and significance tests;
  min-n guard; seeds repeated; pass/fail policy file (YAML) checked in to repo.
- Regression gates in CI: "new adapter must not regress suite X by >Y"; model cards auto-generated (data, metrics,
  limits, licenses, intended use, hardware, carbon/cost).
- LLM-as-judge with calibration + position-bias control; human review queue UI for sampled outputs.
- Robustness: prompt perturbation, long-context, adversarial/jailbreak probes, hallucination checks, bias slices.
- Memorization/PII leakage probe on tuned model (canaries, extraction attack).

## E. Serving and deploy (P1)
- Export: merged HF, GGUF (llama.cpp), AWQ/GPTQ, ONNX; adapter hot-swap (multi-LoRA) on vLLM/llama.cpp.
- One-click deploy: llama.cpp server, vLLM, TGI, SageMaker endpoint, ECS/EKS Helm chart, Bedrock custom import.
- OpenAI-compatible endpoint, streaming, batching, speculative decoding, prompt cache; health, readiness, metrics.
- Serving benchmark: tok/s, TTFT, p50/p95, VRAM, cost per 1M tokens, quality parity vs HF (token agreement %).
- Canary/shadow rollout + rollback of adapters; A/B between model versions with eval gates.

## F. Platform, security, ops (P1)
- Auth (OIDC/SSO, API tokens), RBAC, multi-user/projects/quotas, audit log (who ran/exported what), secret mgmt.
- Durable job queue (SQLite->Postgres/Redis), cancel/retry/priorities, GPU scheduler with VRAM-aware packing.
- Experiment tracking: MLflow/W&B/TensorBoard export; artifact registry (S3) with lineage data->run->model->eval->deploy.
- Observability: OpenTelemetry traces, Prometheus metrics, structured logs, cost tracking per run, budget alarms (exists in TF).
- Security hygiene: dependency pinning/lockfile, pip-audit, SAST, signed releases, SLSA provenance, threat model doc,
  safe pickle handling (torch.load weights_only, safetensors only), path-traversal tests, rate limits.
- IaC: Terraform modules (exists) + CDK option, remote state, per-env, GitHub OIDC (no static keys), cost estimate in PR.
- Compliance posture: data residency options, retention policy, model/data provenance for audits, license checks.

## G. Product and UX (P2)
- Desktop shell (Tauri) + hosted web mode; project workspace; run comparison view (loss curves, eval diffs, samples).
- Dataset browser/editor, prompt playground with side-by-side tuned/base, eval report as shareable HTML/PDF.
- Cost/time estimator before run; "why slow?" profiler panel (launch-bound vs memory-bound vs data-bound).
- Plugin API (datasets, tasks, kernels, exporters); templates/recipes gallery (SQL, extraction, support-bot, classifier).
- Docs site, tutorials, example repos, recorded demo, honest limitations page; community (issues, discussions, roadmap).

## H. Proof and credibility (P0 for adoption)
- Public, reproducible benchmark vs Axolotl/Unsloth/LLaMA-Factory: speed, peak VRAM, final quality, setup time, on
  4GB/8GB/24GB GPUs; publish scripts + raw results; update per release. Be honest where we lose; fix what we can.
- Case studies with real datasets; third-party reproduction; test matrix across OS/GPU/Python/torch versions.
- Semantic versioning, changelog, deprecation policy, LTS; >80% coverage on core; fuzz/property tests for parsers.

## I. Known gaps in the current build (fix list)
- QLoRA only tested on 360M (needs a model that does not fit fp16, e.g. 1.5B-3B on 4GB).
- HF generate ~10 tok/s; GGUF/llama.cpp path in progress (step 3).
- Estimator calibrated on one GPU/model; needs calibration table or on-device probe run (measure real peak, cache it).
- Scratch-trained model is a toy; keep as educational mode, de-emphasize.
- UI untested in a real browser; no auth; single job; scratch `runs/ft` path hard-coded in server for FT chat.
- JAX only a probe; Triton only RMSNorm fwd; no backward kernels.
- NaN-safe JSON events fixed for FT; scratch trainer may still emit NaN? (check).
- ft eval general gate (alpaca-style prompts) is arbitrary for narrow tasks; make eval suite configurable per task.
- Exact-match is case-insensitive for string literals; add execution accuracy with synthetic rows.
- llama.cpp binaries pinned at b11380 (sha256-verified); re-verify when bumping.

## K. From user questions (decide placement later)
- Memory-for-big-models ladder on one small GPU: checkpointing (have) -> chunked CE -> 8-bit/paged optim -> LISA
  (train random layer subset per step) -> layer/CPU/disk offload (ZeRO-Offload/Infinity style; slow but fits).
  Pipeline-parallel layer splitting only helps with >1 device. Layer-wise greedy/local-loss training hurts quality; skip.
- Serving backend abstraction: llama.cpp (default on Windows/laptop; has continuous batching via --parallel) and vLLM
  (PagedAttention + continuous batching + prefix cache + multi-LoRA; needs Linux/WSL2/Docker). Also SGLang/TGI.
  Benchmark under CONCURRENT load (1/8/32 clients); continuous batching only shows up with concurrency.
- UI status: built + HTTP-smoke-tested, never clicked through in a browser; add Playwright e2e test.

## L. Hardware topology, monitoring, portability, offload (from user; candidate roadmap steps 8-11)
- Topology probe (pynvml + nvidia-smi topo): per-GPU name/VRAM/CC/bf16/PCIe gen+width/NVLink, CPU cores, RAM size+speed,
  disk type/free/throughput (NVMe vs HDD), OS. Planner picks strategy from it: 1 GPU (now) | N identical -> DDP
  (throughput) or FSDP (fit) | mixed GPUs -> warn; DDP is bound by slowest GPU + smallest VRAM, so default to best
  single GPU or uneven split (llama.cpp tensor-split for inference) | low VRAM + big RAM -> offload mode.
- Windows has no NCCL (gloo only, slow) and weak DeepSpeed/vLLM: multi-GPU training = Linux/WSL2/Docker; say so in doctor.
- Calibrated on-device probe run (measure real peak mem + tok/s, cache per GPU+model) instead of formula-only estimates.
- Monitoring: Prometheus + Grafana provisioned via docker-compose. GPU: DCGM exporter (Linux) / nvidia_gpu_exporter
  (Windows); host: node_exporter / windows_exporter; app: tinyforge /metrics (loss, tok/s, step, ETA, VRAM, spill flag,
  OOM-recovery count, eval gates). Ship dashboards as JSON. Keep a built-in lightweight panel for users without Grafana.
- "Why slow?" bottleneck panel: util% vs power vs PCIe vs dataloader wait vs disk IO (we saw 45% util = launch-bound).
- Cross-OS: pure-Python core, pathlib everywhere, no .ps1/.sh-only logic; CI matrix ubuntu+windows(+macOS CPU);
  checkpoints are portable (pt/safetensors); capability flags per OS (triton/bnb/flash-attn/vllm) with fallbacks;
  docker parity; WSL2 bridge docs. Test: train on Windows, resume on Linux, same loss curve within tolerance.
- Offload mode (small GPU, big RAM, time no object): DeepSpeed ZeRO-Offload/Infinity or FSDP CPU offload, layer streaming,
  paged/8-bit optimizers, NVMe offload. Planner shows ETA and RAM need (full FT 7B Adam ~112GB states). Expect 10-100x
  slower, PCIe-bound; require resume-safe checkpoints for multi-day runs; laptop thermal/power warnings.
  Honest limits: Windows support poor -> Linux/WSL2; test before claiming.

## M. Production-readiness gaps (developer audit; P0 before any real deploy)
BLOCKERS: no git repo / no history / no LICENSE; torch.load(weights_only=False) in train.load_model + resume paths
(arbitrary code exec on untrusted checkpoint -> switch to safetensors/weights_only=True, adapter via safetensors);
no auth/TLS/rate-limit/CORS policy on API; jobs in an in-memory dict (lost on restart, single process, no persistence);
no lockfile/pinned deps (uv.lock/pip-tools), no pip-audit/SAST/secret-scan in CI; UI never browser-tested; user-supplied
base_model id can pull arbitrary HF repos (allowlist + never trust_remote_code); no config via env/12-factor; logs are
prints not structured; no graceful shutdown of child jobs; CI never actually run; Dockerfile never built.
SOON: API versioning + OpenAPI contract tests; DB (SQLite->Postgres) + migrations; request IDs, structured JSON logs,
OpenTelemetry; /healthz vs /readyz; SLOs + alerts + runbooks; backup/DR for artifacts (S3 versioning exists);
release pipeline (semver tags, changelog, signed images, SBOM, SLSA); environments dev/stage/prod with OIDC deploy;
load tests (k6/locust) for serving; coverage gate; e2e Playwright; chaos tests (kill job, full disk, OOM, spot reclaim).
GOVERNANCE: model cards + lineage registry; license compliance (base model + dataset licenses surfaced and gated);
PII/GDPR handling + retention + right-to-delete for datasets/runs; audit log; threat model doc; vuln disclosure policy.
EXISTING-GOOD: warnings/diagnostic codes, atomic checkpoints, path-traversal guard on scratch run name, IMDSv2,
no-inbound SG, encrypted S3, budgets, non-root container, planned OOM recovery.

## N. Infra/security production standards (Docker, K8s, AWS) - user wants this lens
CURRENT INFRA IS DEV-GRADE: one EC2 in the DEFAULT VPC, public subnet, no ALB/TLS/WAF, no ECR repo in TF, AWS-managed
KMS key (not CMK), no CloudTrail/GuardDuty/Config, no state-lock table, TF not scanned, Dockerfile single-stage unpinned.
DOCKER: multi-stage, pin base by digest, slim runtime, .dockerignore, non-root (have), read-only rootfs + drop caps +
no-new-privileges, HEALTHCHECK (have), no secrets/weights baked in (mount/pull at start), reproducible builds with
locked deps, Trivy/Grype scan gate, SBOM (syft), cosign sign + verify at admission, separate train/serve images,
buildx cache, GPU image matrix (cu124/cu13), ECR lifecycle + immutable tags + scan-on-push.
KUBERNETES (EKS): Helm chart or Kustomize; NVIDIA GPU Operator/device plugin; GPU node groups with taints/tolerations;
Karpenter for GPU spot + consolidation; training as Jobs queued by Kueue/Volcano (gang scheduling, priorities, quotas);
serving as Deployment (vLLM/llama.cpp) behind Service+Ingress(ALB, TLS via ACM/cert-manager); HPA/KEDA on queue depth or
tokens/s custom metrics; readiness/liveness/startup probes (model load is slow); resource requests/limits (nvidia.com/gpu);
PodDisruptionBudget; topology spread; PodSecurity "restricted"; NetworkPolicy default-deny; IRSA / Pod Identity (no static
keys); External Secrets -> Secrets Manager; PVC via EFS/FSx-Lustre/S3 mountpoint for datasets+checkpoints; kube-prometheus
+ DCGM exporter + Grafana; OPA/Kyverno policies (signed images only, no :latest, no privileged); Argo CD GitOps; cluster
upgrades policy; spot interruption handling (checkpoint + node termination handler).
AWS: private subnets + NAT/VPC endpoints (S3, ECR, SSM, STS, Logs), custom VPC via module, multi-AZ; KMS CMKs w/ rotation;
S3 policy deny non-TLS + Block Public Access (have) + Object Lock for model artifacts; ECR; Secrets Manager; IAM least
privilege + permission boundaries + SCPs, GitHub OIDC for CI; ALB + WAF + Cognito/OIDC auth; CloudTrail (org), GuardDuty,
Security Hub, Config rules, Inspector for ECR/EC2; CloudWatch/AMP+AMG; AWS Backup; Budgets (have) + cost anomaly + tags;
Terraform: S3+DynamoDB(or native) state lock, modules, envs/accounts, tflint+checkov+tfsec in CI, plan-in-PR with cost
estimate (infracost), drift detection; SageMaker option for managed training/endpoints; landing zone / multi-account
(prod/stage/dev/security/log-archive); DR: multi-region artifact replication, RPO/RTO documented.
APP SECURITY for LLM: authN/Z, per-tenant isolation of datasets/adapters, prompt-injection + output filtering at serving
edge, rate limits/quotas, request size limits, model/adapter integrity (hash+sign), safetensors only, no trust_remote_code,
egress allowlist for model downloads, audit trail.

## O. Multi-server / datacenter scale (user requirement; shapes ARCHITECTURE EARLY)
ARCH DECISION: split control plane (API, UI, planner, registry, eval gates, audit) from workers (node agent + launcher).
Declarative JobSpec (YAML: model, data, method, resources, parallelism, checkpoints, eval gates) -> Launcher backends:
local subprocess (have) | torchrun/elastic | Slurm (sbatch/srun) | Kubernetes (Job/Kueue/Volcano, PyTorchJob/Kubeflow, Ray)
| SageMaker. Node agent reports topology (GPUs, NVLink, NIC/IB/EFA, RAM, local NVMe) + health to control plane.
TRAINING AT SCALE: don't rebuild -> wrap/standardise TorchTitan/Megatron-LM/DeepSpeed/accelerate-FSDP2/Axolotl-style
backends; our value = planner + preflight + eval gates + tracking. Planner chooses DP/FSDP/TP/PP/EP + micro-batch +
grad-accum + activation ckpt from model size, VRAM, interconnect (NVLink vs PCIe, IB/RoCE/EFA), node count.
Multi-node preflight: NCCL all-reduce/all-gather bandwidth test, GPUDirect RDMA check, clock/driver/CUDA/NCCL version
skew across nodes, NTP skew, ECC/XID errors, thermal throttle, straggler + bad-GPU detection, storage throughput test.
Fault tolerance: elastic restart, async sharded checkpoints (DCP) to shared FS/S3, resume on different world size,
hang watchdog, auto-cordon bad nodes, spot/preemption handling, deterministic data-order resume.
Storage: Lustre/GPFS/FSx/NFS/S3; dataset sharding + streaming (MosaicML-streaming/WebDataset); checkpoint retention policy.
SCHEDULING/MULTI-TENANT: quotas, fair share, priorities, preemption, gang scheduling, chargeback/showback per team,
heterogeneous GPU pools, reservation windows, queue ETA.
DISTRIBUTED INFERENCE: vLLM/TRT-LLM tensor+pipeline parallel, multi-replica + router (KV/prefix-cache-aware), autoscale
on queue depth/tokens/s, multi-LoRA hot-swap, canary adapters, disaggregated prefill/decode (later).
OPS: Prometheus federation/Thanos, DCGM at fleet scale, per-job GPU efficiency (MFU, tokens/s/GPU) dashboards, cost per
run; air-gapped/on-prem install (offline registry, mirrors), mTLS between components, network segmentation, Ansible/
Terraform for bare metal + cloud, Kubernetes Operator for tinyforge, HA control plane (Postgres + stateless API replicas).
VALIDATION: cannot be claimed from a 4GB laptop. Logic tests on CPU (gloo, kind cluster, 2-4 procs) in CI; real
scaling tests on rented multi-node cloud GPUs (publish MFU + scaling efficiency 1->2->4->8 nodes) before any claim.
RISK: scope. Order = JobSpec+Launcher abstraction -> single-node multi-GPU -> multi-node on K8s/Slurm -> fault tolerance.

## P. DIRECTION CHANGE (user, 2026-10-04): Linux-first, k8s-native, kubectl-style CLI (working name `forgectl`)
Decisions: Linux is the target OS (Windows laptop = dev box via WSL2 + CUDA-on-WSL + kind/k3d for testing). Same product
for datacenters and small rigs: ONE control plane + node agent + CLI; single-node users get `forgectl up` (k3s or
no-k8s systemd mode, same agent) so they never need to learn Kubernetes. Stack: Terraform (infra) + Helm (install) +
Kubernetes Operator with CRDs (TrainingJob, ServingEndpoint, Dataset, EvalSuite, ClusterProfile) + Python ML engine
(current tinyforge becomes the worker runtime) + Go for forgectl/agent/operator (client-go, kubebuilder, static binary)
-- CONFIRMED by user: Go for control layer. Go never touches tensors: it launches Python workers (containers) and talks
to them over HTTP/gRPC; training/inference/eval/guardrails stay Python (PyTorch, vLLM, llama.cpp). Define the worker
contract first (JobSpec in, JSON-lines events + exit code out: the current `--json` event stream is the seed of it).
forgectl commands (kubectl feel): get/describe/apply/delete/logs/top/exec for jobs|nodes|endpoints|datasets;
`forgectl doctor [--cluster|--node X]`, `forgectl net plan|check`, `forgectl power`, `forgectl hw`, `forgectl plan -f job.yaml`
(dry-run: fit, ETA, power, cost, network need), `forgectl up/down` (single node), `forgectl cluster bootstrap`.
CLUSTER DOCTOR checks: GPU (XID errors, ECC SBE/DBE, row remap, NVLink errors/lanes, PCIe AER + link gen/width downtrain,
clocks/throttle reasons, persistence mode, MIG state, driver/CUDA/NCCL/container-toolkit versions + skew), CPU/RAM
(EDAC memory errors, NUMA layout + GPU-NUMA affinity, hugepages, swap, THP), storage (SMART/NVMe wear+temp, fs full,
IOPS/throughput test, shared FS mount health), NIC (link speed/duplex, IB/RoCE port state, error counters, MTU, PFC/ECN
config, EFA presence), kernel (sysctls, ulimits, cgroup v2, IOMMU/ACS for GPUDirect, nvidia-peermem), time sync (chrony),
k8s (device plugin, node taints, pods Pending why, PDBs, resource quotas, CNI health, CoreDNS), security posture.
NETWORK PLAN (first-class): declare topology (pods/racks/switches, fabric: Ethernet/RoCE/IB/EFA, oversubscription);
compute per-job traffic model (DP all-reduce bytes/step, TP all-gather, PP p2p, checkpoint write, dataset read, model pull)
-> required bandwidth/latency; ingress plan (ALB/NLB/Ingress-NGINX/Gateway API, TLS, authN, WAF, rate limits, per-tenant
paths); egress plan (default-deny NetworkPolicy/Cilium, allowlist: HF hub, registries, S3 endpoints via VPC endpoints,
proxy/mirror for air-gap); east-west rules (mTLS, namespace isolation); DNS/CIDR/IP-capacity planning (pods per node, IP
exhaustion on EKS); `forgectl net check` runs iperf/NCCL tests + reachability matrix + egress leak test; diagram export (Mermaid/DOT).
POWER/THERMAL: per-GPU/node watts (NVML, DCGM, IPMI/Redfish, PDU/smart-PSU via SNMP/Redfish, RAPL for CPU), power caps
(nvidia-smi -pl) as a scheduler knob, perf/W + tokens/kWh + kWh/run + cost + carbon estimate (grid intensity API), rack
power-budget aware scheduling (don't over-commit a PDU/circuit), thermal throttle alerts, fan/inlet temps, idle-GPU waste
report, UPS/brownout hooks, "run at night/low tariff" scheduling.
HW ISSUE DISPLAY: severity + stable codes + fix text (keep our diagnostic style), node health score, auto-cordon/drain,
RMA-ready evidence bundle (dmesg, nvidia-bug-report, DCGM diag -r 3, SMART), Grafana dashboards + alert rules shipped,
per-job "why slow" (util, power, PCIe/NVLink/IB counters, dataloader wait).
TESTING PLAN: Linux CI w/ kind + fake GPU resources and recorded NVML/DCGM fixtures for logic; real GPU tests on WSL2 box
(1 GPU) and rented cloud GPU nodes (multi-GPU/multi-node); chaos (kill node, drop NIC, fill disk, throttle GPU via -pl);
fault-injection harness feeding synthetic XID/ECC/link errors into doctor to assert diagnoses.
PHASES: 0 baseline (git, safe load, Docker, CI) -> 1 JobSpec + agent + forgectl local (`up`, `plan`, `doctor --node`) ->
2 Helm + Operator + CRDs on k3s/kind -> 3 Terraform EKS/bare-metal modules -> 4 net plan + power -> 5 multi-node training
-> 6 serving (vLLM/llama.cpp) + gates -> 7 multi-tenant/HA/air-gap.

## J. Idea parking lot (append new ideas here; promote to sections when scoped)
- Run "recipes": `tinyforge recipe sql-from-schema` bundles data adapter + hyperparams + eval + gate.
- Auto data-size advisor: learning-curve probe (train on 10/25/50%) to predict returns before full run.
- Adapter marketplace/registry with signed adapters + eval cards.
- Energy/carbon estimate per run; cheapest-cloud finder for jobs that outgrow local GPU.
- "Explain this failure" assistant over metrics + diagnostics codes.
- Federated/private fine-tune mode (data never leaves machine; only adapter exported).
- Differential privacy option (DP-SGD) for sensitive datasets.
- Edge targets: Android/iOS (llama.cpp), Jetson, Raspberry Pi export presets.

### Research reading list (added 2026-10-05; from search snippets, full texts NOT yet read; verify numbers before citing)
- [C] QLoRA, arXiv 2305.14314: 4-bit frozen base + LoRA matches 16-bit finetune; core recipe for 4 GB GPU.
- [step 3 / D] GGUF quant quality: arXiv 2605.19645 (K-Quantization impact: Q8_0 best, Q2_K bad), 2607.08734 (Q6_K-Q4_K avoid sharp
  degradation; perplexity alone insufficient -> parity check must also compare task output), 2402.16775 (quant eval methodology).
- [B / D] Text-to-SQL small models: SLM-SQL 2507.22478 (0.5B 56.87%, 1.5B 67.08% BIRD EX), FINER-SQL 2605.03465 (3B 67.73% BIRD,
  85% Spider; execution-feedback RL). Use as realistic accuracy targets; eval = execution accuracy.
- [E] vLLM/PagedAttention 2309.06180 (KV paging, continuous batching); vLLM vs TGI study 2511.17593 for backend choice.
- [L / O] Reliability: Meta cluster study 2410.21680 (failure taxonomy, MTTF model), Llama 3 infra reliability (DSN-S 2025,
  419 interruptions/54 days), 504-GPU ops analysis 2605.09370 (failure precursors, checkpoint I/O), LLM-PRISM 2604.10390
  (silent data corruption; relevant to checkpoint safety).

## Q. DEPTH PLAN: international-standard engineering (user, 2026-10-05: go DEEPER, not wider)
Rule: no new product areas until the existing ones meet the Definition of Done below. Tool names are from memory, NOT yet
verified for current versions/licences/fit: check each before adopting, and prefer boring, widely used, permissively licensed tools.

DEFINITION OF DONE (every feature, before it counts as done): tests (unit + failure-path, mutation-checked where cheap) -> docs
(how to use + what is NOT covered) -> metrics/log events -> threat-model note -> runbook entry for its failure modes -> works on
Linux (not just this Windows box) -> CI green -> an ADR if it changed a design decision.

### Q1. Supply chain and security (P0: cheap, high trust signal)
- Signing + provenance: Sigstore cosign (sign images/binaries/model artifacts), SLSA build provenance (slsa-github-generator),
  SBOM via syft (SPDX + CycloneDX), verify at admission with Kyverno/policy-controller. OpenSSF model-signing for model files.
- Scanning: Trivy or Grype (images), pip-audit + osv-scanner (Python), govulncheck (Go), gitleaks (secrets, also as pre-commit),
  Checkov/tfsec/tflint-aws (Terraform), kubeconform + kube-linter or Polaris (manifests), Semgrep + CodeQL + Bandit (SAST).
- Hygiene: pin GitHub Actions by commit SHA, Dependabot or Renovate, OpenSSF Scorecard + Best Practices badge, SECURITY.md
  (disclosure policy), CODEOWNERS, branch protection + required checks, signed commits, least-privilege GITHUB_TOKEN, OIDC to cloud.
- Reproducible builds: locked deps (uv lock with torch+CUDA resolved TOGETHER; see the lockfile follow-up), digest-pinned bases (done).
- Threat model: STRIDE doc per component; OWASP ASVS (API), OWASP Top 10 for LLM Apps, MITRE ATLAS (ML-specific attacks).

### Q2. Code quality and testing
- Python: uv, ruff (have), mypy or pyright in strict mode on src/, pytest-cov with a coverage gate, hypothesis (property tests for
  config/spec parsing, data pipeline, exit-code mapping), mutmut (mutation testing on the safety-critical modules), nox for matrix
  runs, pydantic-settings (12-factor config), structlog (JSON logs), safetensors everywhere, Alembic + SQLAlchemy when SQLite -> Postgres.
- Go: golangci-lint, `go test -race` (needs Linux/cgo CI), native fuzzing (`go test -fuzz`) for jobspec/netcheck/agent parsers,
  slog, goreleaser (cross-platform signed releases), controller-runtime + kubebuilder + envtest for the operator.
- Kubernetes/e2e: kind (have) / k3d, Kyverno Chainsaw or Ginkgo for operator e2e, helm-unittest, Testcontainers.
- Resilience: Toxiproxy + Chaos Mesh/LitmusChaos (kill node, drop NIC, fill disk, throttle GPU via power cap), k6 or Locust
  (API + serving load), Playwright (UI, never browser-tested yet), golden-file tests for CLI output, fault-injection fixtures for doctor.
- Pre-commit hooks (ruff, gofmt, gitleaks, actionlint, yamllint, hadolint for Dockerfiles, shellcheck).

### Q3. Observability and operations (SRE practice)
- OpenTelemetry (SDKs + Collector) for traces/metrics/logs across API -> agent -> worker; Prometheus (have) + Alertmanager,
  Grafana (have) + Loki (logs) + Tempo (traces); node_exporter, DCGM exporter on Linux (richer than nvidia_gpu_exporter),
  Node Problem Detector, SLOs as code with Sloth or Pyrra, error budgets, runbooks, blameless postmortem template, DORA metrics.
- Standards: Twelve-Factor, Google SRE workbook practices, CNCF Cloud Native Security whitepaper, CIS Benchmarks (Docker, Kubernetes
  via kube-bench), Pod Security Admission "restricted".

### Q4. Kubernetes and GPU platform (use existing building blocks, do not rebuild)
- Scheduling: Kueue (quotas, fair share, preemption) and/or Volcano (gang scheduling); JobSet (multi-pod jobs); Kubeflow Training
  Operator (PyTorchJob) as a possible backend the operator creates; Ray/KubeRay only if a customer needs it.
- Node/GPU: NVIDIA GPU Operator, device plugin, DCGM, MIG, Node Feature Discovery, NVIDIA Network Operator (RoCE/IB), nccl-tests
  + `dcgmi diag` for preflight; Karpenter for cloud GPU nodes.
- Cluster services: cert-manager (mTLS certs), External Secrets, Cilium + Hubble (NetworkPolicy enforcement + FQDN egress +
  flow visibility; plain NetworkPolicy needs an enforcing CNI), Kyverno or Gatekeeper (signed images only, no :latest, no privileged),
  Argo CD (GitOps), Kustomize + Helm (have), Velero (backup).

### Q5. ML correctness, evals and governance (the moat)
- Evals: lm-evaluation-harness, Inspect AI, promptfoo or DeepEval; execution-accuracy harness for SQL (BIRD/Spider style) per the
  research list; fixed seeds + confidence intervals + paired tests so claims carry error bars; hold-out contamination checks.
- Tracking/data: MLflow (runs/registry) or W&B-compatible export, DVC or lakeFS (data versioning), OpenLineage (lineage events),
  Hugging Face Hub with pinned revisions, model cards (HF standard) generated from eval results, SPDX 3.0 AI profile / CycloneDX
  ML-BOM for AI bills of materials, license checks (ScanCode/ORT) for base models + datasets.
- Safety: Llama Guard or NeMo Guardrails (serving edge), Presidio (PII detection in datasets), garak / PyRIT (red-team scans).
- Frameworks to map controls to: NIST AI RMF, ISO/IEC 42001, EU AI Act (risk tiers, documentation), ISO 27001 / SOC 2 (control
  evidence from audit logs + CI), NIST SSDF (secure development).

### Q6. Training and serving depth (wrap, validate, publish numbers)
- Training: PyTorch FSDP2 + DCP async sharded checkpoints, TorchTitan, DeepSpeed, Accelerate, TRL, Liger-Kernel, torchao
  (quantization), WebDataset or MosaicML Streaming (dataset sharding), nccl-tests (fabric validation).
- Serving: vLLM (continuous batching, prefix cache, multi-LoRA), SGLang, TensorRT-LLM, llama.cpp (have), GenAI-Perf / vLLM
  benchmark scripts under concurrent load (1/8/32 clients).
- Benchmark methodology: follow MLPerf-style reproducibility (fixed workloads, warmup excluded, variance reported, hardware and
  software versions recorded, fair baselines); publish negative results; no claim without the raw JSON committed.

### Q7. Project and community standards
- Licence (Apache-2.0 likely), CONTRIBUTING, CODE_OF_CONDUCT, issue/PR templates, DCO sign-off, SemVer + Keep a Changelog +
  Conventional Commits (drives release notes), ADRs in docs/adr/, C4 diagrams (Mermaid), Diataxis-structured docs site (MkDocs
  Material or Docusaurus), OpenAPI 3.1 + contract tests for the API, versioned JobSpec with a deprecation policy,
  RFC 2119 wording in spec/ documents, public roadmap, release checklist, support matrix (OS, driver, CUDA, k8s versions).

### Q8. Deliberately NOT now (scope guard)
JAX backend, extra Triton kernels, Tauri desktop shell, own trainer/scheduler/inference engine, ROCm/Apple backends, a
marketplace. Revisit only when a design partner asks. Priority inside Q: Q1 + Q2 first (cheap, immediately credible), then Q3
(needed to run anything), then Q4/Q5/Q6 as the controller, serving and multi-node work lands.
