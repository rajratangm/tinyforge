"""Packaging: files the installed package needs at runtime must ship inside it."""

from __future__ import annotations

from pathlib import Path

import tinyforge
from tinyforge import worker

ROOT = Path(__file__).resolve().parents[1]
PKG = Path(tinyforge.__file__).resolve().parent


def test_packaged_jobspec_schema_is_identical_to_the_one_go_embeds():
    packaged = PKG / "jobspec.v1alpha1.schema.json"
    repo = ROOT / "spec" / "jobspec.v1alpha1.schema.json"
    assert packaged.read_bytes() == repo.read_bytes(), "copy the spec/ schema into src/tinyforge/"


def test_worker_uses_the_packaged_schema_not_the_repo_layout(monkeypatch):
    monkeypatch.delenv(worker.SCHEMA_ENV, raising=False)
    assert worker._schema_path() == PKG / "jobspec.v1alpha1.schema.json"


def test_package_data_declares_json_and_ui():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"ui/*"' in text and '"*.json"' in text
    assert (PKG / "ui" / "index.html").exists()
