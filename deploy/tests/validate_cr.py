#!/usr/bin/env python
"""Offline check of TrainingJob custom resources against the generated CRD schema (no cluster needed).

    python deploy/tests/validate_cr.py <file.yaml> [...]    # exit 0 if all valid, 1 if any invalid

Approximates the API server: OpenAPI v3 validation (Draft 4 semantics, which is what CRD schemas use: boolean
exclusiveMinimum, no `const`) plus kubectl's strict field validation for unknown fields, and the root CEL name rule.
It does NOT replicate defaulting, CEL evaluation in general, or admission. The authoritative check is the kind-cluster run
in deploy/README.md. Requires: pyyaml, jsonschema.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from jsonschema import Draft4Validator

CRD = Path(__file__).resolve().parents[1] / "helm" / "tinyforge" / "crds" / "trainingjobs.tinyforge.dev.yaml"


def unknown_fields(schema: dict, value, path: str = "") -> list[str]:
    """Fields present in the object but not declared in the schema (kubectl strict mode rejects these)."""
    found: list[str] = []
    if isinstance(value, dict):
        props = schema.get("properties")
        extra = schema.get("additionalProperties")
        if props is not None:
            for k, v in value.items():
                if k in props:
                    found += unknown_fields(props[k], v, f"{path}.{k}")
                else:
                    found.append(f"{path}.{k}")
        elif isinstance(extra, dict):
            for k, v in value.items():
                found += unknown_fields(extra, v, f"{path}.{k}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            found += unknown_fields(schema["items"], v, f"{path}[{i}]")
    return found


def duplicate_set_items(schema: dict, value, path: str = "") -> list[str]:
    """x-kubernetes-list-type: set forbids duplicates; the API server enforces it, Draft 4 does not know it."""
    found: list[str] = []
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                found += duplicate_set_items(props[k], v, f"{path}.{k}")
    elif isinstance(value, list):
        if schema.get("x-kubernetes-list-type") == "set":
            seen: list = []
            for item in value:
                if item in seen:
                    found.append(f"{path}: duplicate value {item!r} in a set list")
                seen.append(item)
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                found += duplicate_set_items(schema["items"], v, f"{path}[{i}]")
    return found


def check(doc: dict, crd: dict) -> list[str]:
    ver = crd["spec"]["versions"][0]
    root = ver["schema"]["openAPIV3Schema"]
    errors = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
              for e in Draft4Validator(root).iter_errors(doc)]
    for field in unknown_fields({k: v for k, v in root.items() if k != "x-kubernetes-validations"}, doc):
        if not field.startswith((".metadata", ".status", ".apiVersion", ".kind")):
            errors.append(f"unknown field {field} (kubectl strict validation rejects; the server would prune it)")
    errors += duplicate_set_items(root, doc)
    if doc.get("apiVersion") != f"{crd['spec']['group']}/{ver['name']}":
        errors.append(f"apiVersion must be {crd['spec']['group']}/{ver['name']}")
    if doc.get("kind") != crd["spec"]["names"]["kind"]:
        errors.append(f"kind must be {crd['spec']['names']['kind']}")
    for rule in root.get("x-kubernetes-validations", []):
        m = re.search(r"matches\('(.+)'\)", rule["rule"])
        name = doc.get("metadata", {}).get("name")
        if m and name is not None and not re.fullmatch(m.group(1), name):
            errors.append(f"metadata.name: {rule['message']}")
    return errors


def main(paths: list[str]) -> int:
    crd = yaml.safe_load(CRD.read_text(encoding="utf-8"))
    bad = 0
    for p in paths:
        errs = check(yaml.safe_load(Path(p).read_text(encoding="utf-8")), crd)
        print(f"{'INVALID' if errs else 'valid  '} {p}")
        for e in errs:
            print(f"    - {e}")
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
