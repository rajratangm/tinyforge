# deploy/: Kubernetes packaging

Phase 2 of the `forgectl` plan (OBJECTIVES P): the `TrainingJob` CRD and a Helm chart. **There is no controller yet**:
a `TrainingJob` is validated and stored by the API server, but nothing runs it. The chart's controller Deployment is
off by default (`controller.enabled=false`).

```
deploy/
  gen_crd.py                      generates the CRD from spec/jobspec.v1alpha1.schema.json (never edit the CRD by hand)
  helm/tinyforge/                 chart: crds/ (generated), Role/RoleBinding/ServiceAccount, NetworkPolicy, optional controller
  tests/verify.sh                 all checks; `KIND=1` adds a real-cluster run
  tests/validate_cr.py            offline custom-resource validator (no cluster)
  tests/check_netpol.py           renders the chart and asserts the NetworkPolicy safety invariants
  tests/values-network-full.yaml  every network option on, used by the checks above
  tests/fixtures/                 invalid TrainingJobs that must be rejected
```

## Regenerate / check

```
python deploy/gen_crd.py            # rewrite the CRD
python deploy/gen_crd.py --check    # exit 1 if it drifted from the JobSpec schema (run in CI)
bash deploy/tests/verify.sh         # offline checks
KIND=1 bash deploy/tests/verify.sh  # + kind cluster (needs Docker and roughly 2-3 GB of free RAM)
```

## What the CRD can and cannot express

Kubernetes structural schemas are narrower than JSON Schema. How each JobSpec constraint maps:

| JobSpec schema | CRD | Note |
|---|---|---|
| `additionalProperties: false` | not expressible | The API server **prunes** unknown fields. `kubectl apply` (default `--validate=strict`) **rejects** them; clients that skip strict validation get silent pruning. |
| `const: false` (trustRemoteCode) | `enum: [false]` | equivalent |
| `exclusiveMinimum: 0` | `minimum: 0` + `exclusiveMinimum: true` | equivalent |
| `uniqueItems: true` (export) | `x-kubernetes-list-type: set` | equivalent for scalar lists |
| `metadata.name` pattern | CEL rule on the root | Kubernetes alone allows 253-char DNS subdomains; the rule enforces the 63-char DNS label |
| nested `default`s | optional parent objects get `default: {}` | otherwise Kubernetes would not apply child defaults when the parent is absent |
| enum-only nodes with no `type` | type inferred from the enum values | structural schemas require a type on every node |

The converter raises on any JSON Schema keyword it does not know, so a new keyword in the JobSpec can never be silently dropped.

## Security defaults in the chart

Controller pod (when enabled): non-root (uid 65532), read-only root filesystem, all capabilities dropped,
`allowPrivilegeEscalation: false`, seccomp `RuntimeDefault`, requests and limits set, image tag required and `latest` refused.
RBAC is a namespace-scoped `Role` (no ClusterRole, no wildcards, no Secrets). The controller's NetworkPolicy denies all
ingress and allows only DNS egress; **you must set `networkPolicy.controller.egress.apiServer.cidrs`** or the controller cannot
start (the address is cluster specific, so there is no safe default). See "Network policy" below.

## Network policy

The standard behind these templates is `docs/networking.md` (trust zones, port table, allowed flows, what is still planned).

Every component gets its **own default-deny policy** (Ingress and Egress), then only what the values allow:

| Policy | Rendered when | Selects pods labelled |
|---|---|---|
| `*-controller-default-deny` | `networkPolicy.enabled` (default on) | `component=controller` |
| `*-api-default-deny` | `networkPolicy.api.enabled` | `component=api` |
| `*-agent-default-deny` | `networkPolicy.agent.enabled` | `component=agent` |
| `*-worker-default-deny` | `networkPolicy.worker.enabled` | `component=worker` |
| `CiliumNetworkPolicy` per component (FQDN egress) | `networkPolicy.cilium.enabled` | components listed in `cilium.components` |

Only the controller is deployed by this chart. The api, agent and worker policies are **off by default** and select pods by
label, so they have no effect until something with those labels exists.

Rules you can rely on (and `deploy/tests/check_netpol.py` enforces):
- An **empty allow-list renders no rule.** A rule with ports but no `from`/`to` would allow any peer.
- Enabling `controller.metrics` without `metrics.from`, or `cilium` without any FQDN, **fails the render**.
- No rendered rule may allow `0.0.0.0/0`, and every policy is default-deny both ways.

Plain NetworkPolicy has no DNS-name support. For "allow only huggingface.co" use `networkPolicy.cilium` (needs Cilium as the
CNI), an internal mirror, or CIDR allow-lists. Cilium rules are **additive** with the NetworkPolicies, they do not replace them.

```
helm template t deploy/helm/tinyforge -f deploy/tests/values-network-full.yaml   # every option on (documentation-range IPs)
python deploy/tests/check_netpol.py tools/bin/helm.exe                           # safety invariants
```

**Not verified on a real cluster.** NetworkPolicy is enforced only if the cluster's CNI implements it (Calico, Cilium, ...); on a
CNI that does not, these objects are accepted and do nothing. The NCCL/rendezvous port range under `worker.collective` is a
placeholder. `check_netpol.py` checks the rendered YAML, not enforcement.

## Helm and CRDs

Helm installs the CRD from `crds/` on first install but **never upgrades or deletes it**. After changing the JobSpec,
regenerate the CRD and apply it yourself: `kubectl apply -f deploy/helm/tinyforge/crds/`.

## Tools

`tools/bin/` (git-ignored) holds `helm`, `kubectl`, `kind`. Helm was downloaded from get.helm.sh and its sha256 verified
against the published checksum before use.
