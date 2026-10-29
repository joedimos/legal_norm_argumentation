import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from src import cli
from src.audit_bundle import create_bundle, verify_bundle
from src.certificates import certify
from src.obstruction_study import planted
from src.review_planning import plan_review
from src.temporal import atom

SCHEMAS = {
    "vle-certified-result/v1": "certified-result",
    "vle-check-result/v1": "check-result",
    "vle-obstruction-report/v1": "obstruction-report",
    "vle-obstruction-timeline-report/v1": "obstruction-timeline-report",
    "vle-metamorphic-report/v1": "metamorphic-report",
    "vle-review-plan-result/v1": "review-plan",
    "vle-geometry-report/v1": "geometry-report",
    "vle-phase5-error/v1": "phase5-error",
}


def execute(monkeypatch, capsys, tmp_path, command, request, extra=()):
    f = tmp_path / "input.json"
    f.write_text(json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["vle", command, "--input", str(f), *extra])
    with pytest.raises(SystemExit) as code:
        cli.main()
    out = capsys.readouterr()
    r = json.loads(out.out)
    jsonschema.validate(r, json.loads(Path("schemas/" + SCHEMAS[r["schema"]] + ".schema.json").read_text()))
    return code.value.code, r, out.err


def test_all_new_cli_schemas_and_offline_bundle_verification(monkeypatch, capsys, tmp_path):
    query = {"property": {"op": "true"}, "horizon": 1}
    code, r, _ = execute(
        monkeypatch, capsys, tmp_path, "certify", query, extra=["--bundle", str(tmp_path / "cert.zip")]
    )
    assert code == 0 and verify_bundle((tmp_path / "cert.zip").read_bytes())["valid"]
    assert execute(monkeypatch, capsys, tmp_path, "check-certificate", r["certificate"])[0] == 0
    assert execute(monkeypatch, capsys, tmp_path, "obstruction", planted(3, 0, "triangle"))[0] == 0
    corpus = json.loads(Path("benchmarks/metamorphic-v1.json").read_text())
    assert execute(monkeypatch, capsys, tmp_path, "metamorphic", corpus, extra=["--provider", "fake"])[0] == 0
    review = json.loads(Path("benchmarks/review-planning-v1.json").read_text())["cases"][0]["request"]
    assert execute(monkeypatch, capsys, tmp_path, "plan-review", review)[0] == 0
    geometry = {
        "schema": "vle-geometry/v1",
        "horizon": 1,
        "interpretations": [{"id": "p", "property": atom("p").to_dict()}],
    }
    assert execute(monkeypatch, capsys, tmp_path, "geometry", geometry, extra=["--exact", "--topology"])[0] == 0
    assert (
        execute(monkeypatch, capsys, tmp_path, "geometry", geometry, extra=["--sample", "32", "--seed", "23"])[0] == 0
    )
    assert execute(monkeypatch, capsys, tmp_path, "obstruction", {**planted(3, 0, "triangle"), "max_states": 1})[0] == 1
    assert execute(monkeypatch, capsys, tmp_path, "certify", {})[0] == 2
    assert execute(monkeypatch, capsys, tmp_path, "geometry", [])[0] == 2


def test_obstruction_cycle_is_deterministic_across_python_hash_seeds():
    script = 'from src.obstruction import obstruction; from src.obstruction_study import planted; import json; print(json.dumps(obstruction(planted(4,0,"triangle")),sort_keys=True))'
    outputs = []
    for seed in ("0", "1", "23"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(p.stdout)
    assert outputs[0] == outputs[1] == outputs[2]


def test_restriction_projection_is_transitive_and_compatible_sections_glue():
    # Every satisfying global Boolean row agrees with both restrictions.
    rows = [{"p": p, "q": q, "r": r} for p in (False, True) for q in (False, True) for r in (False, True)]
    from src.obstruction import restrict_section as project

    for row in rows:
        assert project(project(row, ["p", "q"]), ["q"]) == project(row, ["q"])
        left, right = project(row, ["p", "q"]), project(row, ["q", "r"])
        assert left["q"] == right["q"] and {**left, **right} == row


def test_family_review_world_adapter_uses_existing_contract_without_approval():
    family = json.loads(Path("examples/annotation-study.json").read_text())["units"][0]["family"]
    family.pop("behavior", None)
    family["observations"] = [{"response:disclose": True}]
    r = plan_review({"schema": "vle-review-plan/v1", "family": family, "questions": []})
    assert r["status"] == "already_decided" and r["review_findings"] and r["review_status"] == "unreviewed"
    assert not r["decision_authorized"]


def test_certificate_cache_budgets_and_unknown_do_not_become_compliance():
    c = certify({"property": {"op": "true"}, "horizon": 1})["certificate"]
    archive, _ = create_bundle({"nested.json": {"certificates": [c, c]}})
    assert verify_bundle(archive)["valid"]
    with pytest.raises(ValueError):
        verify_bundle(archive, expected_certificate_query_sha256s=["bad"])


def test_late_deadline_hits_clear_obstruction_witnesses_and_review_optimality(monkeypatch):
    from src import obstruction as ob
    from src import review_planning as rp
    from src.phase4 import BoundHit

    original = ob.check_certificate

    def late_checker(*args, **kwargs):
        checked = original(*args, **kwargs)
        # A cooperative deadline can expire during the independent replay.
        monkeypatch.setattr(ob.Budget, "tick", lambda self: (_ for _ in ()).throw(BoundHit("timeout")))
        return checked

    with monkeypatch.context() as patch:
        patch.setattr(ob, "check_certificate", late_checker)
        r = ob.obstruction(planted(3, 0, "triangle"))
        assert r["status"] == "unknown" and r["certificate"] is None
    monkeypatch.undo()
    from src import review

    original_order = review.suggested_review_order

    def late_order(*args):
        prompts = original_order(*args)
        monkeypatch.setattr(rp.Budget, "tick", lambda self: (_ for _ in ()).throw(BoundHit("timeout")))
        return prompts

    monkeypatch.setattr(review, "suggested_review_order", late_order)
    req = json.loads(Path("benchmarks/review-planning-v1.json").read_text())["cases"][0]["request"]
    r = rp.plan_review(req)
    assert r["status"] == "unknown" and not r["optimal_in_scope"]
    assert r["plan"] is None and r["certificates"] == [] and r["review_prompts"] == []


def test_report_schemas_reject_unknown_with_partial_success():
    from copy import deepcopy

    from src.geometry import geometry

    request = {
        "schema": "vle-geometry/v1",
        "horizon": 1,
        "interpretations": [{"id": "p", "property": atom("p").to_dict()}],
    }
    report = geometry({**request, "max_states": 1})
    schema = json.loads(Path("schemas/geometry-report.schema.json").read_text())
    jsonschema.validate(report, schema)
    bad = deepcopy(report)
    bad["distances"] = [[0]]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    report = certify({"property": {"op": "true"}, "horizon": 1})
    report["checker"]["valid"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, json.loads(Path("schemas/certified-result.schema.json").read_text()))
