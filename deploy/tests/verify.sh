#!/usr/bin/env bash
# Verify the Kubernetes packaging. Run from the repo root (Git Bash on Windows, or any shell on Linux):
#     bash deploy/tests/verify.sh              # offline checks only (no cluster, needs ~no RAM)
#     KIND=1 bash deploy/tests/verify.sh       # also create a kind cluster and test against a real API server
# Needs: python with pyyaml+jsonschema (PY=...), helm (tools/bin/helm[.exe] or on PATH). KIND=1 also needs docker, kind, kubectl.
set -u
cd "$(dirname "$0")/../.."

PY="${PY:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY="${PY_FALLBACK:-.venv/bin/python}"; [ -x "$PY" ] || PY=python
find_tool() { for c in "tools/bin/$1.exe" "tools/bin/$1" "$1"; do command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }; done; }
HELM="$(find_tool helm)"; KUBECTL="$(find_tool kubectl)"; KIND_BIN="$(find_tool kind)"
CHART=deploy/helm/tinyforge
fail=0
ok()   { echo "PASS  $1"; }
bad()  { echo "FAIL  $1"; fail=1; }
must_pass() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$d"; else bad "$d"; fi; }
must_fail() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then bad "$d (expected a failure)"; else ok "$d"; fi; }

[ -n "$HELM" ] || { echo "helm not found (see deploy/README.md)"; exit 2; }
# Without these imports the Python checks would crash, and a crash counts as "rejected" in the must_fail checks below:
# the negative tests would pass for the wrong reason. Fail loudly instead.
"$PY" -c "import yaml, jsonschema" 2>/dev/null || {
  echo "FAIL  python needs pyyaml and jsonschema (PY=$PY). Install them, or point PYTHONPATH at a directory that has them."
  exit 2
}

echo "== CRD generation"
must_pass "CRD matches spec/jobspec.v1alpha1.schema.json (no drift)" "$PY" deploy/gen_crd.py --check

echo "== Helm chart"
must_pass "helm lint (defaults)"        "$HELM" lint "$CHART"
must_pass "helm lint (controller on)"   "$HELM" lint "$CHART" --set controller.enabled=true --set controller.image.tag=v0.1.0
must_pass "helm template (defaults)"    "$HELM" template t "$CHART"
must_fail "controller refuses tag 'latest'" "$HELM" template t "$CHART" --set controller.enabled=true --set controller.image.tag=latest
must_fail "controller refuses an empty tag" "$HELM" template t "$CHART" --set controller.enabled=true
if "$HELM" template t "$CHART" | grep -q '^kind: Deployment'; then bad "controller is rendered by default (must be off)"; else ok "controller is off by default"; fi

echo "== Network policies (render + safety invariants; no cluster)"
must_pass "helm lint (all network options on)" "$HELM" lint "$CHART" -f deploy/tests/values-network-full.yaml
must_pass "NetworkPolicy safety checks (default-deny both ways, no open rules, misconfig refused)" "$PY" deploy/tests/check_netpol.py "$HELM"

echo "== Custom resources (offline schema check)"
must_pass "example TrainingJob is valid"            "$PY" deploy/tests/validate_cr.py spec/examples/sql-finetune.yaml
must_fail "invalid method is rejected"              "$PY" deploy/tests/validate_cr.py deploy/tests/fixtures/invalid-method.yaml
must_fail "unknown field is rejected"               "$PY" deploy/tests/validate_cr.py deploy/tests/fixtures/invalid-unknown-field.yaml

if [ "${KIND:-0}" = "1" ]; then
  echo "== kind cluster (real API server). Needs ~2-3 GB free RAM."
  [ -n "$KUBECTL" ] && [ -n "$KIND_BIN" ] && command -v docker >/dev/null 2>&1 || { echo "need docker, kind and kubectl"; exit 2; }
  cleanup() { "$KIND_BIN" delete cluster --name tinyforge-dev >/dev/null 2>&1; }
  trap cleanup EXIT
  "$KIND_BIN" create cluster --name tinyforge-dev --wait 120s >/dev/null 2>&1 && ok "kind cluster created" || { bad "kind cluster create"; exit 1; }
  must_pass "helm install"                         "$HELM" install tinyforge "$CHART" -n tinyforge --create-namespace
  must_pass "CRD established"                      "$KUBECTL" wait --for=condition=Established crd/trainingjobs.tinyforge.dev --timeout=60s
  must_pass "server accepts the example TrainingJob"       "$KUBECTL" apply -f spec/examples/sql-finetune.yaml
  must_fail "server rejects method: lora2"                 "$KUBECTL" apply -f deploy/tests/fixtures/invalid-method.yaml
  must_fail "server rejects an unknown field (strict)"     "$KUBECTL" apply --validate=strict -f deploy/tests/fixtures/invalid-unknown-field.yaml
  "$KUBECTL" get trainingjobs
fi

echo
[ "$fail" = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit "$fail"
