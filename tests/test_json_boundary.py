"""Synthetic/unreviewed boundary regressions; no legal authority is asserted."""

from __future__ import annotations

import json
import sys

import pytest

from src import cli
from src.audit_bundle import canonical_json
from src.json_boundary import MAX_DEPTH, strict_json
from src.model_runtime import strict_json as public_strict_json


@pytest.mark.parametrize(
    "raw",
    [
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e10000}',
        '{"x":-1e10000}',
        '{"x":1,"x":2}',
        '{"x":{"a":1,"a":2}}',
        '{"x":"\\ud800"}',
        '{"\\udfff":1}',
        b'{"x":"\xff"}',
        '{"x":true}'.encode("utf-16"),
        "[" * (MAX_DEPTH + 2) + "0" + "]" * (MAX_DEPTH + 2),
        "[" * 2000 + "0" + "]" * 2000,
    ],
)
def test_malformed_nonfinite_or_unencodable_json_is_rejected(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


@pytest.mark.parametrize(
    "value",
    [
        {"synthetic": True, "review_status": "unreviewed", "text": "Propriété – droit ©"},
        {"emoji": "🧑", "number": 1.25e100},
        [None, True, 1, -2.5, "x"],
    ],
)
def test_valid_unicode_is_preserved_without_hash_drift(value):
    encoded = canonical_json(value)
    assert public_strict_json is strict_json
    assert canonical_json(strict_json(encoded)) == encoded
    assert strict_json(bytearray(encoded)) == value
    assert strict_json(b"\xef\xbb\xbf" + encoded) == value


def test_depth_boundary_and_valid_surrogate_pair():
    raw = "[" * MAX_DEPTH + "0" + "]" * MAX_DEPTH
    assert strict_json(raw) is not None
    assert strict_json('"\\ud83e\\uddd1"') == "🧑"


def test_cli_inline_and_file_share_strict_validation(tmp_path):
    assert cli._load_json('{"synthetic":true}') == {"synthetic": True}
    path = tmp_path / "input.json"
    path.write_text('{"x":1e10000}')
    with pytest.raises(ValueError):
        cli._load_json(str(path))
    with pytest.raises(ValueError):
        cli._load_json('{"x":true,"x":false}')
    path.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="16 MiB"):
        cli._load_json(str(path))


@pytest.mark.parametrize(
    "content", ['{"schema":"invalid","x":1e10000}', '{"schema":"invalid","x":"\\ud800"}', '{"schema":"a","schema":"b"}']
)
def test_research_cli_rejects_before_dispatch_with_serializable_failure(tmp_path, monkeypatch, capsys, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    monkeypatch.setattr(sys, "argv", ["vle-legato", "research", "--input", str(path)])
    with pytest.raises(SystemExit) as code:
        cli.main()
    assert code.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"error"} and report["error"]
