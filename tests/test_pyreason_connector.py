"""Synthetic, unreviewed adapter contracts; no live model or network."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pyreason_connector import PREDICATES, PyReasonConnector, prepare
from src.pyreason_worker import execute


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
        Path(cmd[-1]).write_text(json.dumps({"status": "observed", "annotations": {}}))
    monkeypatch.setattr(subprocess, "run", incomplete)
    report = PyReasonConnector().run_reasoning({"a": {}}, set())
    assert report["status"] == "unknown" and not report["converged"]


def test_recorded_output_is_only_advisory(monkeypatch):
    def recorded(cmd, **kwargs):
        assert kwargs["timeout"] == 60
        Path(cmd[-1]).write_text(json.dumps({"status": "observed", "backend_version": "recorded",
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
    monkeypatch.setattr(worker, "version", lambda name: "fixture")
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
