import json
import sys
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from src import cli
from src.aspic import analyze_aspic
from src.assumptions import assumption_report
from src.audit_bundle import create_bundle, verify_bundle
from src.faithfulness import faithfulness_report
from src.norm_codec import decode_norm
from src.normative_model import Modality, NormInterpretation
from src.redundancy.crosscheck import consensus_report
from src.redundancy.generate import FakeProvider, RuntimeProvider

SCHEMAS = {
    "vle-aspic-result/v1": "aspic-result",
    "vle-assumption-report/v1": "assumption-report",
    "vle-redundancy-report/v1": "redundancy-report",
    "vle-redundancy-generation/v1": "redundancy-generation",
    "vle-ddl-interchange/v1": "ddl-interchange",
    "vle-faithfulness-report/v1": "faithfulness-report",
    "vle-phase4-error/v1": "phase4-error",
}


def validate(result):
    schema = json.loads(Path("schemas/" + SCHEMAS[result["schema"]] + ".schema.json").read_text())
    jsonschema.validate(result, schema)


def aspic_request(**kw):
    return {
        "schema": "vle-aspic/v1",
        "premises": [],
        "rules": [{"id": "r", "antecedents": [], "kind": "defeasible", "conclusion": "p"}],
        **kw,
    }


def assumptions_request(**kw):
    return {
        "schema": "vle-assumptions/v1",
        "expected": {"op": "atom", "name": "p"},
        "horizon": 1,
        "assumptions": [],
        **kw,
    }


def run_cli(monkeypatch, capsys, tmp_path, command, payload=None, extra=()):
    argv = ["vle", command]
    if payload is not None:
        path = tmp_path / "input.json"
        path.write_text(json.dumps(payload))
        argv += ["--input", str(path)]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    with pytest.raises(SystemExit) as code:
        cli.main()
    streams = capsys.readouterr()
    result = json.loads(streams.out)
    validate(result)
    return code.value.code, result, streams.err


def test_every_phase4_cli_output_and_bundle_schema(monkeypatch, capsys, tmp_path):
    code, r, _ = run_cli(monkeypatch, capsys, tmp_path, "aspic", aspic_request(), extra=["--lifting", "weakest"])
    assert code == 0 and r["lifting"] == "weakest"
    bundle = tmp_path / "assumptions.zip"
    code, r, err = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "assumptions",
        assumptions_request(),
        extra=["--backend", "exhaustive", "--max-mcs", "5", "--bundle", str(bundle)],
    )
    assert (
        code == 0
        and verify_bundle(bundle.read_bytes(), expected_manifest_digest=json.loads(err)["bundle_manifest_digest"])[
            "valid"
        ]
    )
    code, r, _ = run_cli(
        monkeypatch, capsys, tmp_path, "assumptions", assumptions_request(), extra=["--bundle", str(bundle)]
    )
    assert code == 2 and r["status"] == "unknown"  # never overwrite an audit artifact
    ddl = {"schema": "vle-ddl/v1", "operation": "segment", "source_id": "synthetic", "text": "Operators must notify."}
    assert run_cli(monkeypatch, capsys, tmp_path, "ddl", ddl)[0] == 0
    assert run_cli(monkeypatch, capsys, tmp_path, "faithfulness")[0] == 0
    text = tmp_path / "source.txt"
    text.write_text("The operator must notify.")
    code, generated, _ = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "redundant-formalize",
        extra=["--source-id", "synthetic", "--text-file", str(text), "--candidates", "2", "--provider", "fake"],
    )
    assert code == 0 and generated["interpretation_set"]["selected_id"] is None
    code, report, _ = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "redundancy-report",
        generated,
        extra=["--bundle", str(tmp_path / "redundancy.zip")],
    )
    assert (
        code == 0
        and report["consensus_supported"]
        and verify_bundle((tmp_path / "redundancy.zip").read_bytes())["valid"]
    )
    incomplete = {**generated, "generation_calls": 3}
    assert run_cli(monkeypatch, capsys, tmp_path, "redundancy-report", incomplete)[0] == 2
    assert run_cli(monkeypatch, capsys, tmp_path, "aspic", aspic_request(max_states=1))[0] == 1
    assert run_cli(monkeypatch, capsys, tmp_path, "aspic", {"schema": "unknown"})[0] == 2


def test_cli_explicit_live_routes_are_fake_only(monkeypatch, capsys, tmp_path):
    import src.redundancy.generate as module

    corpus = json.loads(Path("benchmarks/phase4-v1.json").read_text())
    corpus["cases"] = corpus["cases"][:1]
    formalization = corpus["cases"][0]["formalization"]

    class ConfiguredFake(FakeProvider):
        def __init__(self, config, *, system_prompt=None):
            assert config["provider"] == "vllm" and config["model"] == "fake"
            outputs = (
                [{"verdict": "empty"}]
                if system_prompt and "verdict only" in system_prompt
                else (
                    [formalization]
                    if system_prompt
                    else [NormInterpretation("n", "s", Modality.OBLIGATION, "operator", "notify").to_dict()]
                )
            )
            super().__init__(outputs)

    monkeypatch.setattr(module, "RuntimeProvider", ConfiguredFake)
    assert (
        run_cli(
            monkeypatch,
            capsys,
            tmp_path,
            "faithfulness",
            corpus,
            extra=["--live", "--provider", "vllm", "--model", "fake", "--bundle", str(tmp_path / "faith.zip")],
        )[0]
        == 0
    )
    assert verify_bundle((tmp_path / "faith.zip").read_bytes())["artifact_count"] == 3
    text = tmp_path / "source.txt"
    text.write_text("The operator must notify.")
    assert (
        run_cli(
            monkeypatch,
            capsys,
            tmp_path,
            "redundant-formalize",
            extra=["--source-id", "synthetic", "--text-file", str(text), "--provider", "vllm", "--model", "fake"],
        )[0]
        == 0
    )


def test_schema_mutations_cannot_launder_unknown_or_authorization():
    a = NormInterpretation("a", "s", Modality.OBLIGATION, "s", "a")
    b = NormInterpretation("b", "s", Modality.OBLIGATION, "s", "a")
    r = consensus_report(
        {
            "schema": "vle-redundancy/v1",
            "candidates": [{"norm": a.to_dict()}, {"norm": b.to_dict()}],
            "horizon": 1,
            "max_traces": 1,
        }
    )
    # Force a bounded comparison with one unconstrained action: state cap unknown.
    assert r["status"] == "unknown"
    cases = [
        (analyze_aspic(aspic_request(max_states=1)), "extensions", [["invented"]]),
        (assumption_report(assumptions_request(max_states=1)), "minimality_established", True),
        (r, "consensus_supported", True),
    ]
    for report, key, value in cases:
        validate(report)
        mutated = deepcopy(report)
        mutated[key] = value
        with pytest.raises(jsonschema.ValidationError):
            validate(mutated)
        mutated = deepcopy(report)
        mutated["decision_authorized"] = True
        with pytest.raises(jsonschema.ValidationError):
            validate(mutated)


def test_deadline_expiration_is_unknown_and_discards_acceptance(monkeypatch):
    import src.phase4 as module

    for runner, request in [
        (analyze_aspic, aspic_request()),
        (assumption_report, assumptions_request()),
        (faithfulness_report, json.loads(Path("benchmarks/phase4-v1.json").read_text())),
    ]:
        times = iter([0, 100])
        monkeypatch.setattr(module, "monotonic", lambda clock=times: next(clock))
        result = runner(request)
        assert result["status"] == "unknown" and not result["decision_authorized"]
        validate(result)


def test_asymmetric_contrary_attack_is_not_preference_blocked():
    request = aspic_request()
    request["rules"].append({"id": "s", "antecedents": [], "kind": "defeasible", "conclusion": "q"})
    request["contraries"] = {"p": ["q"]}
    request["preferences"] = [{"preferred": "r", "disfavored": "s", "rationale": "synthetic"}]
    result = analyze_aspic(request)
    assert result["attacks"][0]["successful"]


def test_no_network_guard_is_effective():
    import socket

    import urllib3

    with pytest.raises(AssertionError, match="offline"):
        socket.create_connection(("example.invalid", 443))
    with pytest.raises(AssertionError, match="offline"):
        urllib3.PoolManager().request("GET", "https://example.invalid")


def test_legacy_bundle_format_still_verifies_and_native_codec_rejects_tampering():
    archive, root = create_bundle(
        {"legacy.json": {"schema": "vle-research/v1", "status": "unknown"}}, metadata={"version": "0.6.2"}
    )
    assert verify_bundle(archive, expected_manifest_digest=root)["valid"]
    norm = NormInterpretation("n", "s", Modality.OBLIGATION, "s", "a").to_dict()
    for fields in [
        {"canonical": "invented"},
        {"semantic_hash": "0" * 64},
        {"conditions": [{"name": "p", "negated": "true"}]},
        {"references": [True]},
        {"object": False},
        {"source_norm_id": ""},
        {"unknown_field": True},
    ]:
        with pytest.raises(ValueError):
            decode_norm({**norm, **fields})


def test_runtime_rejects_incomplete_model_and_oversized_prompt_before_transport():
    called = []

    def transport(*a, **kw):
        called.append(True)
        return {"done": False, "model": "fake"}

    provider = RuntimeProvider({"provider": "vllm", "model": "fake"}, transport=transport)
    with pytest.raises(ValueError, match="incomplete"):
        provider.generate("source", seed=0, temperature=0)
    with pytest.raises(ValueError, match="1 MiB"):
        provider.generate("x" * 1100000, seed=0, temperature=0)
    assert len(called) == 1
