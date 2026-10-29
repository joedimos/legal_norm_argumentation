import json
from itertools import combinations
from pathlib import Path

import jsonschema
import pytest

from src.assumptions import assumption_report
from src.temporal import Formula, atom, combine


def a(id, f):
    return {
        "id": id,
        "kind": "background_fact",
        "formula": f.to_dict(),
        "source_span": None,
        "description": "Synthetic unreviewed assumption " + id,
    }


def req(assumptions, expected=None, **kw):
    return {
        "schema": "vle-assumptions/v1",
        "assumptions": assumptions,
        "expected": (expected or Formula("true")).to_dict(),
        "horizon": 1,
        **kw,
    }


def signature(r):
    return (
        r["status"],
        [m["assumption_ids"] for m in r["minimal_corrections"]],
        [m["assumption_ids"] for m in r["minimal_unsatisfiable"]],
    )


def validate(r):
    jsonschema.validate(r, json.loads(Path("schemas/assumption-report.schema.json").read_text()))
    assert not r["decision_authorized"] and not r["legal_certification"]


def test_unique_multiple_and_hard_conflict():
    p, q = atom("p"), atom("q")
    unique = assumption_report(req([a("not-p", combine("not", p)), a("q", q)], p))
    validate(unique)
    assert unique["status"] == "corrections_found"
    assert unique["minimal_corrections"][0]["assumption_ids"] == ["not-p"]
    assert unique["review_prompts"] and unique["review_status"] == "unreviewed"
    multiple = assumption_report(req([a("p", p), a("not-p", combine("not", p))]))
    assert [m["assumption_ids"] for m in multiple["minimal_corrections"]] == [["not-p"], ["p"]]
    conflict = assumption_report(req([], p, hard=[combine("not", p).to_dict()]))
    assert conflict["status"] == "hard_conflict" and not conflict["minimal_corrections"]
    assert conflict["minimal_unsatisfiable"][0]["assumption_ids"] == []
    feasible = assumption_report(req([a("p", p)], p))
    assert feasible["status"] == "expected_feasible" and feasible["minimal_corrections"][0]["assumption_ids"] == []


def test_exhaustive_z3_differential_and_deterministic_order():
    pytest.importorskip("z3")
    p, q = atom("p"), atom("q")
    forms = [p, combine("not", p), q, combine("not", q), combine("or", p, q), combine("next", p)]
    for subset in combinations(range(len(forms)), 3):
        assumptions = [a(str(i), forms[i]) for i in subset]
        r = req(assumptions)
        e = assumption_report(r)
        z = assumption_report(r, backend="z3")
        validate(z)
        assert signature(e) == signature(z)
        assert signature(e) == signature(assumption_report(req(list(reversed(assumptions)))))


@pytest.mark.parametrize("kw", [{"max_states": 1}, {"max_traces": 1}, {"max_mcs": 1}, {"max_mus": 1}])
def test_truncation_suppresses_all_minimality_claims(kw):
    p = atom("p")
    r = assumption_report(req([a("p", p), a("n", combine("not", p))], **kw))
    validate(r)
    assert r["status"] == "unknown" and r["truncated"]
    assert not r["minimality_established"] and not r["minimal_corrections"] and not r["review_prompts"]


def test_source_spans_and_assumption_kinds_and_input_validation():
    item = a("p", atom("p"))
    item["source_span"] = {"source_id": "synthetic", "start": 2, "end": 5, "quote": "abc"}
    for kind in ("predicate_grounding", "interpretation_selection", "horizon_closure", "boolean_completion"):
        item["kind"] = kind
        validate(assumption_report(req([item])))
    for bad in [
        dict(item, kind="guessed"),
        dict(item, source_span={"bad": 0}),
        dict(item, source_span={**item["source_span"], "end": 4}),
    ]:
        with pytest.raises(ValueError):
            assumption_report(req([bad]))
    with pytest.raises(ValueError):
        assumption_report(req([item, item]))
    with pytest.raises(ValueError):
        assumption_report(req([], horizon=True))


def test_unknown_solver_and_replay_failures_are_not_explanations(monkeypatch):
    import src.assumptions_z3 as module

    class UnknownOracle:
        def __init__(self, *a):
            from src.phase4 import BoundHit

            raise BoundHit("z3_unknown:test")

    monkeypatch.setattr(module, "Z3Oracle", UnknownOracle)
    r = assumption_report(req([a("p", atom("p"))]), backend="z3")
    assert r["status"] == "unknown" and not r["minimal_corrections"]


def test_temporal_oracle_encoding_agrees_with_exhaustive():
    pytest.importorskip("z3")
    p, q = atom("p"), atom("q")
    forms = [Formula("false"), Formula("true"), combine("and", p, q), combine("until", p, q)]
    forms += [combine(op, p) for op in ("always", "eventually", "next", "weak_next")]
    for f in forms:
        for horizon in (1, 2):
            r = req([a("candidate", f), a("opposite", combine("not", f))], horizon=horizon)
            e, z = assumption_report(r), assumption_report(r, backend="z3")
            assert signature(e) == signature(z)
