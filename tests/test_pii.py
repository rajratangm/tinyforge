from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tinyforge.cli import app
from tinyforge.pii import mask, redact, scan, summarize


def kinds(text: str) -> list[str]:
    return [f.kind for f in scan(text)]


@pytest.mark.parametrize("text,kind", [
    ("mail me at jane.doe+x@example.co.uk please", "email"),
    ("call +1 415-555-0132 tomorrow", "phone"),
    ("call (415) 555-0132 tomorrow", "phone"),
    ("card 4111 1111 1111 1111 on file", "card"),
    ("card 4111-1111-1111-1111", "card"),
    ("ssn 123-45-6789", "ssn"),
    ("pay to GB82 WEST 1234 5698 7654 32 now", "iban"),
    ("server at 8.8.8.8 responded", "ip"),
    ("key AKIAIOSFODNN7EXAMPLE leaked", "aws_key"),
    ("token ghp_" + "a1B2" * 9, "github_token"),
    ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk", "jwt"),
])
def test_detects(text, kind):
    assert kinds(text) == [kind]


@pytest.mark.parametrize("text", [
    "order 4111 1111 1111 1112 shipped",      # fails Luhn
    "the date 2026-10-07 is fine",             # not a phone
    "rented after '2005-08-23 02:06:01' and 2018-03-17 07:13:53.5",  # timestamps
    "version 1.2.3 and 10.0.0.1 and 192.168.1.1 and 127.0.0.1",  # private/loopback IPs, short versions
    "ssn 000-12-3456 and 666-12-3456 and 900-12-3456",           # invalid SSN ranges
    "GB00 WEST 1234 5698 7654 32",             # bad IBAN checksum
    "call 1234567890123456789",                # bare digit run, no separators
    "nothing personal here, just prose.",
])
def test_no_false_positives_on_lookalikes(text):
    assert scan(text) == []


def test_secrets_are_flagged_secret_and_pii_is_not():
    fs = scan("a@b.com and AKIAIOSFODNN7EXAMPLE")
    assert {(f.kind, f.secret) for f in fs} == {("email", False), ("aws_key", True)}


def test_redact_replaces_in_place_and_keeps_surroundings():
    t = "Email a@b.com or call +1 415-555-0132."
    assert redact(t) == "Email [EMAIL] or call [PHONE]."


def test_card_is_not_double_reported_as_phone():
    assert kinds("4111 1111 1111 1111") == ["card"]


def test_kinds_filter_limits_pii_but_never_secrets():
    t = "a@b.com 123-45-6789 AKIAIOSFODNN7EXAMPLE"
    assert sorted(kinds(t)) == ["aws_key", "email", "ssn"]
    got = [f.kind for f in scan(t, kinds=["ssn"])]
    assert sorted(got) == ["aws_key", "ssn"]


def test_mask_never_prints_the_full_value():
    assert "@" not in mask("jane.doe@example.com").replace("**", "")[2:-2]
    assert mask("short") == "*****"


def test_summarize_counts_and_masks():
    counts, hit, ex = summarize(["a@b.com", "clean", "x@y.org and 123-45-6789"])
    assert counts["email"] == 2 and counts["ssn"] == 1 and hit == 2
    assert all("@" not in e[2:-2] for e in ex["email"])


def test_cli_scan_reports_and_exits_nonzero_on_secrets(tmp_path):
    f = tmp_path / "d.jsonl"
    rows = [{"messages": [{"role": "user", "content": "hi a@b.com"}, {"role": "assistant", "content": "ok"}]},
            {"text": "clean row"}]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    res = CliRunner().invoke(app, ["data", "pii-scan", str(f), "--json"])
    out = json.loads(res.stdout)
    assert out["rows"] == 2 and out["rows_with_findings"] == 1 and out["by_kind"] == {"email": 1}
    assert res.exit_code == 0
    f.write_text(json.dumps({"text": "AKIAIOSFODNN7EXAMPLE"}), encoding="utf-8")
    assert CliRunner().invoke(app, ["data", "pii-scan", str(f), "--json"]).exit_code == 1


def test_ingest_policies_flag_redact_drop_and_secrets_always_dropped(tmp_path):
    from tinyforge.etl import ingest

    filler = " ".join(f"word{i}" for i in range(30))
    (tmp_path / "a.txt").write_text(f"Contact jane@example.com today. {filler}\n")
    (tmp_path / "b.txt").write_text(f"Key AKIAIOSFODNN7EXAMPLE leaked. {filler} other\n")
    (tmp_path / "c.txt").write_text(f"Plain text only. {filler} third\n")
    flag, m_flag, _ = ingest([tmp_path], pii_policy="flag")
    assert len(flag) == 2 and m_flag["dropped_secret"] == 1
    assert any("jane@example.com" in c.text for c in flag)
    red, _, _ = ingest([tmp_path], pii_policy="redact")
    assert any("[EMAIL]" in c.text for c in red) and not any("@" in c.text for c in red)
    drop, m_drop, _ = ingest([tmp_path], pii_policy="drop")
    assert len(drop) == 1 and m_drop["dropped_pii"] == 1


def test_ft_data_prepare_policies(tmp_path, monkeypatch):
    import transformers

    from tinyforge import ft_data

    class Tok:
        chat_template = "{{ messages }}"  # prepare() now refuses tokenizers without one

        def apply_chat_template(self, m, tokenize=True, return_dict=False):
            return [1, 2, 3]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    rows = [{"instruction": f"question number {i}", "output": f"answer {i} mail bob{i}@example.com"}
            for i in range(60)]
    rows.append({"instruction": "leak", "output": "key AKIAIOSFODNN7EXAMPLE"})
    src = tmp_path / "s.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    meta, _ = ft_data.prepare(str(src), tmp_path / "f", "x", limit=100, val_pct=20, pii_policy="redact")
    out = (tmp_path / "f" / "train.jsonl").read_text() + (tmp_path / "f" / "val.jsonl").read_text()
    assert "@example.com" not in out and "[EMAIL]" in out and "AKIA" not in out
    assert meta["pii_redacted"] == 60 and meta["pii_by_kind"]["aws_key"] == 1
    meta2, _ = ft_data.prepare(str(src), tmp_path / "g", "x", limit=100, val_pct=20, pii_policy="drop")
    assert meta2["train"] + meta2["val"] == 0 and meta2["pii_dropped"] == 60
