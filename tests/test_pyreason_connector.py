"""Synthetic, unreviewed adapter contracts; no live model or network."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from src.pyreason_connector import BACKEND_VERSION, PREDICATES, PyReasonConnector, digest, prepare
from src.pyreason_worker import execute


@pytest.fixture(autouse=True)
def validate_every_report(monkeypatch):
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/pyreason-report.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    run = PyReasonConnector.run_reasoning

    def checked(*args, **kwargs):
        report = run(*args, **kwargs)
        validator.validate(report)
        return report

    monkeypatch.setattr(PyReasonConnector, "run_reasoning", checked)


@pytest.mark.parametrize("confidence", [[0, 1], [1, 1], [0, 0], [0.3, 0.7]])
def test_safe_identifiers_and_determinism(confidence):
    args = {"a):[1,1], injected(x)": {"confidence": confidence}, "b": {}}
    one, names = prepare(args, set(), set(), 2)
    two, _ = prepare(dict(reversed(list(args.items()))), set(), set(), 2)
    assert one == two
    assert names["a):[1,1], injected(x)"] == "n0"
    assert one["nodes"]["n1"] == [0, 1]


@pytest.mark.parametrize("confidence", [[float('nan'), 1], [1, 0], [-1, 0], [True, 1], "bad"])
def test_invalid_input_never_runs(monkeypatch, confidence):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not invoke engine"))
    report = PyReasonConnector().run_reasoning({"x": {"confidence": confidence}}, set())
    assert report["status"] == "unknown"
    assert not report["accepted"] and not report["decision_authorized"]


def test_timeout_and_incomplete_results(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    report = PyReasonConnector().run_reasoning({"a": {}}, set(), timeout_seconds=1)
    assert report["bound_hit"] and report["status"] == "unknown"
    def incomplete(cmd, **kwargs):
        Path(cmd[-1]).write_text(json.dumps({"status": "observed", "annotations": {},
            "backend_version": BACKEND_VERSION, "input_sha256": digest(json.loads(Path(cmd[-2]).read_text()))}))
    monkeypatch.setattr(subprocess, "run", incomplete)
    report = PyReasonConnector().run_reasoning({"a": {}}, set())
    assert report["status"] == "unknown" and not report["converged"]


def test_recorded_output_is_only_advisory(monkeypatch):
    def recorded(cmd, **kwargs):
        assert kwargs["timeout"] == 60
        Path(cmd[-1]).write_text(json.dumps({"status": "observed", "backend_version": BACKEND_VERSION,
            "input_sha256": digest(json.loads(Path(cmd[-2]).read_text())),
            "annotations": {p: {"n0": [0, 1]} for p in PREDICATES}}))
    monkeypatch.setattr(subprocess, "run", recorded)
    report = PyReasonConnector().run_reasoning({"original id": {}}, set())
    assert report["accepted"] == {"original id": [0, 1]}
    assert report["status"] == "observed_within_bound"
    assert report["converged"] is False and report["decision_authorized"] is False


def test_worker_upstream_shape_and_unknowns():
    calls = []
    pr = SimpleNamespace(settings=SimpleNamespace(), reset=lambda: calls.append("reset"),
        reset_rules=lambda: calls.append("rules"), reset_settings=lambda: calls.append("settings"),
        load_graph=lambda g: None, Fact=lambda text, **kw: (text, kw),
        Rule=lambda text, **kw: (text, kw), add_fact=lambda f: calls.append(f),
        add_rule=lambda r: calls.append(r), reason=lambda **kw: SimpleNamespace(
            get_dict=lambda: {0: {"n0": TraceDict()}}))
    payload, _ = prepare({"a": {"confidence": [0.4, 0.8]}}, set(), set(), 2)
    output = execute(payload, pr)
    assert calls[:3] == ["reset", "rules", "settings"]
    assert output["accepted"]["n0"] == [0.4, 0.8]
    assert output["defeated"]["n0"] == [0, 1]
    assert pr.settings.store_interpretation_changes


@pytest.mark.parametrize("horizon", [0, True, 101])
def test_bounds(horizon):
    report = PyReasonConnector().run_reasoning({}, set(), timesteps=horizon)
    assert report["status"] == "unknown"


def test_worker_receipt_and_failure(tmp_path, monkeypatch):
    import sys

    from src import pyreason_worker as worker
    source, target = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text('{}')
    monkeypatch.setattr(sys, "argv", ["worker", str(source), str(target)])
    monkeypatch.setitem(sys.modules, "pyreason", SimpleNamespace())
    monkeypatch.setattr(worker, "version", lambda name: BACKEND_VERSION)
    monkeypatch.setattr(worker, "execute", lambda payload, pr: {})
    worker.main()
    assert json.loads(target.read_text())["status"] == "observed"
    def fail(*args):
        raise RuntimeError("private source text")
    monkeypatch.setattr(worker, "execute", fail)
    worker.main()
    assert json.loads(target.read_text()) == {"status": "unknown"}


def test_schema_and_bounds(monkeypatch):
    import jsonschema
    report = PyReasonConnector().run_reasoning({"a": {}}, {("a", "missing")})
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/pyreason-report.schema.json").read_text())
    jsonschema.validate(report, schema)
    assert report["status"] == "unknown"


class TraceDict(dict):
    """Upstream InterpretationDict overrides items, but not dict.get."""
    def items(self):
        return [("accepted", (0.4, 0.8))]


def receipt(cmd, **kwargs):
    """Deterministic recorded worker, preserving input annotations exactly."""
    payload = json.loads(Path(cmd[-2]).read_text())
    result = {"status": "observed", "backend_version": BACKEND_VERSION, "input_sha256": digest(payload),
              "annotations": {p: {node: [0, 1] for node in payload["nodes"]} for p in PREDICATES}}
    result["annotations"]["accepted"] = payload["nodes"]
    return result


@pytest.mark.parametrize("mutation", ["version", "request", "missing", "interval", "lost_input", "unexpected_node"])
def test_incompatible_receipts_fail_closed(monkeypatch, mutation):
    def broken(cmd, **kwargs):
        result = receipt(cmd)
        if mutation == "version":
            result["backend_version"] = "3.7.0"
        elif mutation == "request":
            result["input_sha256"] = "0" * 64
        elif mutation == "missing":
            del result["annotations"]["strengthened"]["n0"]
        elif mutation == "interval":
            result["annotations"]["defeated"]["n0"] = [1, 0]
        elif mutation == "lost_input":
            result["annotations"]["accepted"]["n0"] = [0, 1]
        else:
            result["annotations"]["defeated"]["invented"] = [1, 1]
        Path(cmd[-1]).write_text(json.dumps(result))
    monkeypatch.setattr(subprocess, "run", broken)
    report = PyReasonConnector().run_reasoning({"a": {"confidence": [1, 1]}}, set())
    assert report["status"] == "unknown"
    assert all(report[p] == {} for p in PREDICATES)


@pytest.mark.parametrize("which", ["horizon", "arguments", "edges", "output"])
def test_all_resource_caps_are_reported(monkeypatch, which):
    def oversized(cmd, **kwargs):
        assert which == "output", "input limits must block before subprocess execution"
        Path(cmd[-1]).write_bytes(b"x" * 2_000_001)
    monkeypatch.setattr(subprocess, "run", oversized)
    arguments = {f"a{i}": {} for i in range(257)} if which == "arguments" else {"a": {}}
    attacks = [("a", "a")] * 4097 if which == "edges" else set()
    report = PyReasonConnector().run_reasoning(arguments, attacks, timesteps=101 if which == "horizon" else 2)
    assert report["status"] == "unknown" and report["bound_hit"]
    assert report["error"].endswith("bound")
    assert report["bounds"]["max_output_bytes"] == 2_000_000


@pytest.mark.parametrize("arguments,attacks", [({"a": None}, set()), ({"a": {}}, {"aa"}),
                                             ({"a": {}}, [("a", [])])])
def test_malformed_inputs_never_invoke_engine(monkeypatch, arguments, attacks):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    report = PyReasonConnector().run_reasoning(arguments, attacks)
    assert report["status"] == "unknown" and not report["accepted"]


def test_worker_rejects_unqualified_version_before_import(tmp_path, monkeypatch):
    import builtins
    import sys

    from src import pyreason_worker as worker
    target = tmp_path / "output.json"
    monkeypatch.setattr(sys, "argv", ["worker", "not-read.json", str(target)])
    monkeypatch.setattr(worker, "version", lambda name: "3.7.0")
    original_import = builtins.__import__

    def no_pyreason(name, *args, **kwargs):
        assert name != "pyreason", "unsupported backend must not execute import-time code"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyreason)
    worker.main()
    assert json.loads(target.read_text()) == {"status": "unknown", "error": "unsupported_version"}


def test_public_wrappers_preserve_symbolic_results_and_timeout(monkeypatch):
    from src.argumentation import LegalArgument, OptimizedAAF
    from src.originality import LogicBasedLegalReasoner
    from src.originality_annotator import OriginalityAnnotator

    timeouts = []

    def recorded(cmd, **kwargs):
        timeouts.append(kwargs["timeout"])
        Path(cmd[-1]).write_text(json.dumps(receipt(cmd)))

    monkeypatch.setattr(subprocess, "run", recorded)
    aaf = OptimizedAAF([LegalArgument("first", confidence=(0, 1)), LegalArgument("second")],
                       attacks={("first", "second")}, supports={("second", "first")})
    before = aaf.export_to_dict()
    report = aaf.run_pyreason(2, timeout_seconds=300)
    assert report["accepted"]["first"] == [0, 1]
    assert aaf.export_to_dict() == before
    text = "The work reflects the author's personality and free creative choices."
    reasoner = LogicBasedLegalReasoner()
    baseline = reasoner.analyze_legal_case(text)
    analysis = OriginalityAnnotator(reasoner).annotate_originality(text, True, pyreason_timeout_seconds=120)
    assert timeouts == [300, 120]
    assert analysis["aaf"] == baseline["aaf"]
    assert analysis["pyreason_results"]["status"] == "observed_within_bound"
    assert analysis["pyreason_eval"] == baseline["pyreason_eval"]
    assert analysis["pyreason_eval"]["decision_authorized"] is False


def test_hashes_bind_rules_graph_and_argument_identity(monkeypatch):
    def recorded(cmd, **kwargs):
        Path(cmd[-1]).write_text(json.dumps(receipt(cmd)))
    monkeypatch.setattr(subprocess, "run", recorded)
    connector = PyReasonConnector()
    first = connector.run_reasoning({"a": {}, "b": {}}, {("a", "b")})
    reordered = connector.run_reasoning({"b": {}, "a": {}}, [("a", "b"), ("a", "b")])
    renamed = connector.run_reasoning({"other_a": {}, "other_b": {}}, {("other_a", "other_b")})
    assert first == reordered
    assert first["input_sha256"] == renamed["input_sha256"]
    assert first["argument_map_sha256"] != renamed["argument_map_sha256"]
    assert first["rule_set_sha256"] == renamed["rule_set_sha256"]


def test_installation_paths_select_the_supported_backend():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["optional-dependencies"]["pyreason"] == [f"pyreason=={BACKEND_VERSION}"]
    requirements = [line.strip() for line in (root / "requirements-pyreason.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    assert requirements == [".[pyreason]"]
