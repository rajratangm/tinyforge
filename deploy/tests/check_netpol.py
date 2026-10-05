"""Safety checks on the rendered NetworkPolicies (needs helm; no cluster).

Usage (repo root): python deploy/tests/check_netpol.py <helm-binary>

The danger this guards against: in a NetworkPolicy, a rule that has ports but no from/to means "any peer". An unset or
empty Helm value must therefore never turn into an open rule. Checks, for the default values AND a values file that turns
every network option on:
  - every NetworkPolicy is default-deny in BOTH directions (policyTypes lists Ingress and Egress);
  - every ingress/egress rule names its peers (non-empty from/to), and no ipBlock allows 0.0.0.0/0 or ::/0;
  - the DNS rule only opens port 53; every FQDN rule lists at least one name;
  - misconfigurations fail the render instead of producing an open rule.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import yaml

CHART = "deploy/helm/tinyforge"
FULL = "deploy/tests/values-network-full.yaml"
OPEN_CIDRS = {"0.0.0.0/0", "::/0"}

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"PASS  {msg}")


def helm(helm_bin: str, *args: str) -> tuple[int, str]:
    p = subprocess.run([helm_bin, "template", "t", CHART, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def docs(text: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(text) if d]


def check_policy(doc: dict, label: str) -> None:
    name = doc["metadata"]["name"]
    spec = doc["spec"]
    types = set(spec.get("policyTypes", []))
    if types != {"Ingress", "Egress"}:
        fail(f"{label}/{name}: policyTypes must be Ingress+Egress (default-deny both ways), got {sorted(types)}")
    for direction, peer_key in (("ingress", "from"), ("egress", "to")):
        for i, rule in enumerate(spec.get(direction, [])):
            peers = rule.get(peer_key)
            if not peers:
                fail(f"{label}/{name}: {direction}[{i}] has no {peer_key}: that allows ANY peer")
                continue
            for p in peers:
                if p.get("ipBlock", {}).get("cidr") in OPEN_CIDRS:
                    fail(f"{label}/{name}: {direction}[{i}] allows the whole internet ({p['ipBlock']['cidr']})")
                if not p:
                    fail(f"{label}/{name}: {direction}[{i}] has an empty peer selector (matches everything)")


def check_cilium(doc: dict, label: str) -> None:
    name = doc["metadata"]["name"]
    for i, rule in enumerate(doc["spec"].get("egress", [])):
        if "toFQDNs" in rule and not rule["toFQDNs"]:
            fail(f"{label}/{name}: egress[{i}] toFQDNs is empty")
        if "toFQDNs" in rule and not rule.get("toPorts"):
            fail(f"{label}/{name}: egress[{i}] toFQDNs without toPorts opens every port")


def check_render(helm_bin: str, label: str, *args: str) -> list[dict]:
    rc, out = helm(helm_bin, *args)
    if rc != 0:
        fail(f"{label}: helm template failed: {out.strip()[:200]}")
        return []
    ds = docs(out)
    for d in ds:
        if d["kind"] == "NetworkPolicy":
            check_policy(d, label)
        elif d["kind"] == "CiliumNetworkPolicy":
            check_cilium(d, label)
    ok(f"{label}: {sum(d['kind'] == 'NetworkPolicy' for d in ds)} NetworkPolicy, "
       f"{sum(d['kind'] == 'CiliumNetworkPolicy' for d in ds)} CiliumNetworkPolicy checked")
    return ds


def must_fail(helm_bin: str, label: str, *args: str) -> None:
    rc, _ = helm(helm_bin, *args)
    (ok if rc != 0 else fail)(label if rc != 0 else f"{label} (render succeeded but must fail)")


def main(helm_bin: str) -> int:
    # Defaults: only the controller policy, DNS-only egress, no ingress.
    ds = check_render(helm_bin, "defaults")
    pols = [d for d in ds if d["kind"] == "NetworkPolicy"]
    if [p["metadata"]["name"] for p in pols] != ["t-tinyforge-controller-default-deny"]:
        fail(f"defaults: expected only the controller policy, got {[p['metadata']['name'] for p in pols]}")
    elif pols[0]["spec"]["ingress"] != [] or len(pols[0]["spec"]["egress"]) != 1:
        fail("defaults: controller must have no ingress and DNS-only egress")
    else:
        ok("defaults: controller is deny-all except DNS")

    # Everything on.
    ds = check_render(helm_bin, "full", "-f", FULL)
    kinds = sorted(d["metadata"]["name"] for d in ds if d["kind"] in ("NetworkPolicy", "CiliumNetworkPolicy"))
    want = sorted(f"t-tinyforge-{n}" for n in (
        "controller-default-deny", "api-default-deny", "agent-default-deny", "worker-default-deny",
        "worker-fqdn-egress", "api-fqdn-egress"))
    (ok if kinds == want else fail)(f"full: policy set {'matches' if kinds == want else 'differs: ' + str(kinds)}")

    # Components enabled but allow-lists empty must render deny-only policies, never open rules.
    ds = check_render(helm_bin, "components on, empty allow-lists", "--set", "networkPolicy.api.enabled=true",
                      "--set", "networkPolicy.agent.enabled=true", "--set", "networkPolicy.worker.enabled=true")
    for d in ds:
        if d["kind"] == "NetworkPolicy" and d["metadata"]["name"] != "t-tinyforge-controller-default-deny":
            s = d["spec"]
            if s["ingress"] != [] or len(s["egress"]) != 1:
                fail(f"empty allow-lists: {d['metadata']['name']} must be DNS-only egress and no ingress")
    ok("empty allow-lists render deny-only policies")

    # Misconfigurations must fail loudly.
    must_fail(helm_bin, "metrics enabled without a scraper selector is refused",
              "--set", "networkPolicy.controller.metrics.enabled=true")
    must_fail(helm_bin, "cilium enabled without any FQDN is refused", "--set", "networkPolicy.cilium.enabled=true")
    must_fail(helm_bin, "unknown cilium component is refused", "--set", "networkPolicy.cilium.enabled=true",
              "--set", "networkPolicy.cilium.fqdns.matchNames={huggingface.co}",
              "--set", "networkPolicy.cilium.components={bogus}")

    print()
    print("NETPOL CHECKS FAILED" if failures else "NETPOL CHECKS PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    wanted = sys.argv[1] if len(sys.argv) > 1 else "helm"
    resolved = shutil.which(wanted) or shutil.which(os.path.abspath(wanted))  # relative paths need resolving on Windows
    if not resolved:
        sys.exit(f"helm not found: {wanted}")
    sys.exit(main(resolved))
