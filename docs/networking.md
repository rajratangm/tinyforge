# Networking standard

How tinyforge components talk to each other and to the outside world, what is enforced today, and what is still a plan.
Written 2026-10-05 against the code in this repository. Other tracks were changing `infra/terraform`, `internal/agent`
and `src/tinyforge/netsec.py` while this was written, so re-check those rows before relying on them.

Status legend: **Implemented** (in the repo and tested), **Partial** (some of it works; the gap is stated), **Planned** (design only, nothing built).
Nothing here has been verified on a real multi-node cluster. There is no cluster-level test yet (see section 10).

## 1. Trust zones

Traffic should only ever flow inward one zone at a time, never skipping a zone.

| Zone | Contains | Trust | Status |
|---|---|---|---|
| Z0 Users / internet | browsers, `forgectl` on a laptop, CI | untrusted | n/a |
| Z1 Ingress | load balancer / ingress controller, WAF | terminates TLS, authenticates | Planned (no ingress manifests exist) |
| Z2 Control plane | tinyforge API, operator/controller, job store | trusted, holds credentials | Partial: API exists, operator does not |
| Z3 Node agents | `forgectl agent` on every GPU server | trusted, talks to hardware | Partial: agent exists, job execution is a stub |
| Z4 Workers | training / inference pods or processes | **least trusted** (run user data and models) | Planned: no operator launches workers yet |
| Z5 Storage / registry / model hub | S3, shared FS, container registry, Hugging Face | external or semi-trusted | n/a |
| Z6 Observability | Prometheus, Grafana, GPU exporter | read-only view of Z2-Z4 | Implemented locally (compose), loopback only |

Rules: Z4 never initiates connections to Z2 or Z3. Z4 may reach Z5 only through an explicit allow-list. Z0 never reaches Z3 or Z4 directly.

## 2. Ports and protocols (components that exist today)

| Component | Port | Protocol | Default bind | Auth | Encryption | Source of truth |
|---|---|---|---|---|---|---|
| tinyforge API (`tinyforge serve`) | 8000 | HTTP/1.1 (REST + SSE) | `127.0.0.1` (CLI default) | bearer token (`TINYFORGE_API_TOKEN`) | none by default. `netsec.resolve_serve` adds TLS/mTLS flags but `cli.py` does not call it yet | `src/tinyforge/cli.py`, `server.py`, `netsec.py` |
| forgectl node agent | 7070 | HTTP/1.1 | `127.0.0.1:7070` | bearer token and/or client certificates; refuses non-loopback without credentials | optional TLS 1.2+ and mTLS (`internal/agent/tls.go`, with tests from another track; I did not run them) | `internal/agent/agent.go`, `tls.go` |
| Prometheus | 9090 | HTTP | `127.0.0.1` (compose) | none | none | `infra/monitoring/docker-compose.yml` |
| Grafana | 3000 | HTTP | `127.0.0.1` (compose) | admin password from `.env`; anonymous off | none | `infra/monitoring/docker-compose.yml` |
| NVIDIA GPU exporter | 9835 | HTTP | `127.0.0.1` (compose) | none | none | `infra/monitoring/docker-compose.yml` |
| llama.cpp `llama-server` (benchmarks only) | 8099 | HTTP | loopback | none | none | `tools/gguf_parity.py` |
| Training rendezvous (torchrun `--master_port`) | 29500 by default | TCP | cluster network | none | none | **Planned**, not used yet |
| NCCL collectives | ephemeral TCP ports, or IB/RoCE/EFA | TCP / RDMA | cluster fabric | none | none (fabric must be private) | **Planned**, section 8 |
| Kubernetes API (controller to API server) | 443 or 6443 | HTTPS | in-cluster | ServiceAccount token | TLS | `deploy/helm` (egress allow-list) |

Container image: `Dockerfile` runs `tinyforge serve --host 0.0.0.0`, so the container listens on all interfaces over plain HTTP.
When `netsec.resolve_serve` is wired into the CLI, that command will be refused without TLS or `--insecure-http`: update the image
command at the same time.

## 3. Allowed flows (default-deny; anything not listed is blocked)

| # | From | To | Port | Auth | Encrypted | Enforced by | Status |
|---|---|---|---|---|---|---|---|
| F1 | user / `forgectl` | ingress | 443 | OIDC or token | TLS 1.2+ | LB / ingress | Planned |
| F2 | ingress | tinyforge API | 8000 | bearer token | mesh mTLS or in-pod TLS | NetworkPolicy `api.ingressFrom` | Partial (policy exists, off by default) |
| F3 | tinyforge API / controller | node agent | 7070 | token or client cert | mTLS | NetworkPolicy `agent.ingressFrom` | Partial |
| F4 | controller | Kubernetes API server | 443/6443 | ServiceAccount | TLS | NetworkPolicy `controller.egress.apiServer` | Partial (you must set the CIDR) |
| F5 | worker | model hub (HF) | 443 | none / HF token | TLS | NetworkPolicy CIDRs or Cilium FQDN | Partial (CIDR/FQDN rules, untested on a cluster) |
| F6 | worker | container registry | 443 | pull secret | TLS | NetworkPolicy `worker.egress.registry` | Partial |
| F7 | worker | object storage / shared FS | 443, 2049 | IAM / mount | TLS (S3), none (NFS) | NetworkPolicy `worker.egress.storage` | Partial |
| F8 | worker | worker (same namespace) | rendezvous + NCCL range | none | none | NetworkPolicy `worker.collective` | Planned: port range is a placeholder, untested |
| F9 | Prometheus | API, agent, controller `/metrics` | 8000, 7070, 8080 | bearer token | as above | NetworkPolicy `metricsFrom` / `metrics.from` | Partial |
| F10 | Prometheus | GPU exporter | 9835 | none | none | compose network (loopback host binds) | Implemented (local only) |
| F11 | everything | cluster DNS | 53 UDP/TCP | n/a | n/a | every policy allows it explicitly | Implemented in the chart |

Not allowed, by design: worker to API/agent/controller; worker to the Kubernetes API; any component to `0.0.0.0/0` on a
non-443 port; ingress straight to workers; one tenant's namespace to another's.

## 4. North-south (user to platform)

- Terminate TLS 1.2 or newer (1.3 preferred) at the ingress / load balancer, not in each component. Certificates are
  issued and rotated there (cert-manager or ACM). **Planned**: no ingress manifest exists yet.
- WAF and rate limiting at the edge. The API already rate-limits repeated authentication failures and caps request bodies
  (1 MiB), sends hardening headers and a CSP, and allows CORS only for an explicit origin list
  (`src/tinyforge/netsec.py`, wired in `server.py`): **Implemented** at the application layer, but that is a second line
  of defence, not a substitute for the edge.
- Authentication at the edge (OIDC) plus the API's own bearer token. Tokens are compared in constant time.
- Never expose the node agent, Prometheus, Grafana or the GPU exporter to the internet.
- Single-node installs (`forgectl up`, planned) may use the built-in TLS. The API's CLI wiring for it is not done yet: **Partial**.

## 5. East-west (inside the cluster)

- Default-deny NetworkPolicy per component, in both directions (**Implemented** in the Helm chart, default-deny for the
  controller is on by default; api/agent/worker policies are opt-in because those pods are not deployed by this chart yet).
- Namespace isolation: one tenant or trust group per namespace. Plain NetworkPolicy cannot say "same job only", so a job's
  workers can reach every worker in their namespace when `worker.collective.enabled` is on.
- mTLS between components: **Partial**. The agent has TLS 1.2+ and mTLS in code (`internal/agent/tls.go`); the API has the code in
  `netsec.py` but it is not reachable from the CLI yet. No service mesh, no certificate issuance or rotation between components.
- NetworkPolicy only works if the cluster's CNI enforces it (Calico, Cilium, and similar). On a CNI that does not, the
  objects are accepted and silently do nothing. Check this on every new cluster before trusting the policies.
- Safety rule in the chart: an empty allow-list renders **no** rule. A rule with ports but no `from`/`to` would mean "any peer".
  `deploy/tests/check_netpol.py` fails the build if any rendered rule is open or allows `0.0.0.0/0`.

## 6. Egress

Default for every workload: DNS only. Everything else is an explicit allow-list.

| Destination | How it is expressed | Status |
|---|---|---|
| Kubernetes API server | CIDR of the API endpoints (`kubectl get endpoints kubernetes`) | Partial: no safe default, the controller cannot start until set |
| Model hub, registry, storage | CIDR allow-lists, or a Cilium FQDN policy (`networkPolicy.cilium`) | Partial: untested against a real cluster |
| S3 | VPC gateway endpoint, so traffic stays on the AWS network | Planned (`infra/terraform` has no VPC endpoints today) |
| Everything else | denied | Implemented in the chart policies |

Plain NetworkPolicy has no DNS-name support. Options in order of preference: an internal mirror or proxy inside the cluster
(also the air-gap answer: mirror models, images and Python wheels, then allow only the mirror), an egress gateway, or Cilium
`toFQDNs` (**Partial**: template provided, off by default, needs Cilium).
**Air-gap mode** (offline registry, model mirror, no internet egress at all): **Planned**.

AWS dev infra today (`infra/terraform/main.tf`, may be changing): one EC2 instance in the account's **default VPC** unless
`vpc_id`/`subnet_id` are set, security group with **no inbound** rules, egress on 443 to `0.0.0.0/0`, IMDSv2 required.
That is dev-grade: production wants private subnets, VPC endpoints (S3, ECR, STS, logs, SSM), and egress narrowed from the open 443 rule.

## 7. DNS

- Pods use cluster DNS; every policy allows UDP and TCP 53 to it only. The selector is configurable (`networkPolicy.dns`) for
  NodeLocal DNSCache or OpenShift.
- Split-horizon names for internal services; no component should depend on a public resolver.
- Cilium FQDN rules need DNS visibility, which the template enables via a DNS proxy rule.
- Status: cluster DNS egress **Implemented**; the rest **Planned**.

## 8. Address planning and IP capacity (Planned)

- Plan CIDRs before the first cluster: node subnets, pod CIDR, service CIDR must not overlap each other, the corporate
  network, or a peered VPC. Changing them later usually means rebuilding the cluster.
- Pod capacity per node on EKS with the default VPC CNI is bounded by ENI limits:
  `max pods = ENIs x (IPv4 addresses per ENI - 1) + 2` for the instance type (look the values up per instance type; do not guess).
  Large GPU nodes also run many system pods, so check headroom. Prefix delegation raises the ceiling at the cost of larger
  address blocks per node.
- GPU training pods are usually few per node (one per GPU), so the real IP pressure comes from the sidecars and system daemons, and from
  subnets sized too small for autoscaled node counts. Size subnets for the maximum node count plus the pods per node.
- Reserve address space for the future multi-node fabric (section 9) separately from the pod network.

## 9. Cluster fabric for multi-node training (Planned)

Nothing in this section is built or measured. It is the model the planned `forgectl net plan` would compute from.

**Traffic model.** N = GPUs in the group, S = bytes in the tensor being synchronised, b = micro-batch, s = sequence length, h = hidden size.

| Pattern | Bytes per GPU per step (each direction) | Notes |
|---|---|---|
| Data-parallel all-reduce (ring) | `2(N-1)/N x S`, S = trainable params x bytes per gradient element | For LoRA only the adapter gradients are synchronised, so S is tiny and the network is rarely the limit |
| FSDP / ZeRO-3 | roughly `3(N-1)/N x S_params` (two all-gathers plus a reduce-scatter) | Approximate: about 1.5x the plain all-reduce volume |
| Tensor-parallel (per layer, per micro-batch) | 4 all-reduces of `b x s x h x bytes`, each `2(N-1)/N` times that | Latency-bound and needs the fastest links (NVLink); keep TP inside one node |
| Pipeline-parallel | `b x s x h x bytes` per micro-batch per stage boundary, forward and backward | Point-to-point, tolerates slower links |
| Checkpoint write | params x state bytes per parameter (Adam in mixed precision is about 16, which gives 112 GB for 7B, matching the planning note in OBJECTIVES) | Burst, size to the storage network |
| Dataset read | tokens per second x bytes per token x workers | Usually small unless streaming raw media |

Required bandwidth is `bytes per step / (target step time x fraction of the step you allow communication to take)`.
Arithmetic only, not a measurement: a 7B-parameter model with bf16 gradients has S = 14 GB; ring all-reduce over N = 8 moves
`2 x 7/8 x 14 = 24.5 GB` per GPU per direction per step, which takes 0.49 s on a 50 GB/s link if nothing overlaps with compute.

**Fabric choice.**

| Fabric | Use | Notes |
|---|---|---|
| Ethernet with TCP | small clusters, LoRA, inference | cheapest; NCCL over sockets is the slowest option |
| RoCEv2 | datacenter Ethernet with RDMA | needs lossless configuration (PFC/ECN) and consistent MTU end to end |
| InfiniBand | large synchronous training | best latency; separate fabric to manage |
| AWS EFA | AWS GPU instances | libfabric plugin for NCCL; needs EFA-enabled instance types and security group rules allowing all traffic within the group |

Checklist: non-blocking (1:1) or a stated oversubscription ratio for all-reduce heavy jobs, jumbo frames only if every hop supports them
(a mismatched MTU black-holes large packets), a dedicated interface for collective traffic (NCCL `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`),
GPUDirect RDMA where hardware allows, and NTP/chrony in sync across nodes.

**Policy caveat.** The torchrun rendezvous port is set by the launcher (default 29500), and NCCL opens additional ports.
The chart's `worker.collective.portRange` (29400-29999) is a placeholder that must be matched to the real launcher and verified.
On RDMA fabrics (IB, RoCE, EFA) NCCL traffic bypasses the pod network, so Kubernetes NetworkPolicy does not govern it: isolate
that fabric physically or with partitioning (for example IB partition keys, or dedicated VLANs).

## 10. Network observability and testing

Scrape targets and what exists:

| Signal | Source | Status |
|---|---|---|
| GPU utilisation, memory, temperature, power, PCIe link width | GPU exporter 9835 | Implemented (local) |
| Queue depth, jobs by status, live loss and tok/s | tinyforge API `/metrics` | Implemented |
| Node health, GPU count, doctor findings | agent `/metrics` | Implemented |
| NIC throughput, errors, drops | node_exporter netdev / ethtool collectors | Planned |
| InfiniBand / RoCE port counters, link state | node_exporter infiniband collector, DCGM | Planned |
| Flow logs and policy verdicts | VPC Flow Logs, Cilium Hubble | Planned |
| NCCL communication time per step | worker `/metrics` | Planned |

Testing plan:

- **`forgectl net check` is planned, not built.** It would run a reachability matrix against section 3, an egress leak test (confirm
  blocked destinations really are blocked), a DNS check, an MTU path check, and `iperf3` and `nccl-tests` for bandwidth.
- Built today: `deploy/tests/check_netpol.py` renders the chart and asserts every policy is default-deny in both directions, no
  rule is open or allows `0.0.0.0/0`, and unsafe configurations fail instead of rendering. It runs under `deploy/tests/verify.sh`.
  It tests the rendered YAML only. **It does not prove a cluster enforces the policies.**
- Planned: a kind cluster with a CNI that enforces NetworkPolicy (Calico or Cilium; do not assume the default kind CNI does)
  running positive and negative connectivity tests for every row of section 3; chaos tests (drop a NIC, fill a link); real multi-node
  scaling runs on rented GPU nodes before any performance claim.

## 11. Honest gap list

1. No TLS on the API from the CLI (the code exists but is not wired), agent TLS is untested by me, and none on monitoring
   (monitoring is loopback-only for that reason). No certificate issuance or rotation.
2. No ingress, WAF or edge authentication manifests; no service mesh or mTLS between components in practice.
3. NetworkPolicies are rendered and linted, never enforced on a real cluster.
4. No VPC endpoints, private subnets or narrowed egress in the AWS infra.
5. No worker launcher, so the worker policy and the NCCL port range are untested guesses.
6. No NIC, InfiniBand or flow-level monitoring; no `forgectl net plan` or `net check`.
7. Air-gap mode, mirrors and egress gateway are designs only.
