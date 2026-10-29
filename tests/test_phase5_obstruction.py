import json
from pathlib import Path

import jsonschema
import pytest

from src.checker import check_certificate
from src.obstruction import gyo_acyclic, obstruction
from src.obstruction_study import equal, planted, run_study
from src.temporal import atom


def test_planted_obstructions_and_independent_refutations():
    for size in (3, 4):
        request = planted(size, 0, "triangle")
        r = obstruction(request)
        jsonschema.validate(r, json.loads(Path("schemas/obstruction-report.schema.json").read_text()))
        assert r["status"] == "cyclic_obstruction" and len(r["minimal_family"]) == size
        assert all(p["consistent"] for p in r["pair_checks"])
        assert check_certificate(r["certificate"])["valid"]
        assert obstruction(request) == r


def test_exhaustive_and_z3_agree_and_acyclic_shortcut_crosschecks():
    pytest.importorskip("z3")
    for size in (3, 4):
        req = planted(size, 2, "reference_loop")
        assert obstruction(req)["status"] == obstruction(req, backend="z3")["status"]
    p, q, r = atom("p"), atom("q"), atom("r")
    request = {
        "schema": "vle-obstruction/v1",
        "horizon": 1,
        "assume_reviewed_demo": True,
        "cells": [{"id": "a", "property": equal(p, q).to_dict()}, {"id": "b", "property": equal(q, r).to_dict()}],
    }
    out = obstruction(request)
    assert out["acyclic_cover"] and out["status"] == "globally_consistent"
    assert obstruction({**request, "crosscheck_shortcut": False})["status"] == out["status"]
    assert check_certificate(out["certificate"])["valid"]


def test_acyclic_pair_sat_claim_is_refuted_not_hidden():
    study = run_study()
    assert study["F1"]["passed"] == 32
    bad = study["acyclic_counterexample"]
    assert bad["acyclic_cover"] and all(p["consistent"] for p in bad["pair_checks"])
    assert bad["status"] == "higher_order_conflict" and bad["cycle"] == []
    assert study["F3"]["status"] == "not_run_missing_reviewed_data" and study["F3"]["obstructions"] is None


def test_bounds_review_required_and_gyo():
    req = planted(3, 0, "triangle")
    r = obstruction({**req, "max_states": 1})
    assert r["status"] == "unknown" and r["certificate"] is None
    assert obstruction({**req, "assume_reviewed_demo": False})["status"] == "unknown"
    assert gyo_acyclic([{"p"}, {"p", "q"}, {"q"}])
    assert not gyo_acyclic([{"p", "q"}, {"q", "r"}, {"p", "r"}])


def test_amendment_obstruction_lifetimes_are_censored_and_bounded():
    req = planted(3, 0, "triangle")
    timeline = {
        "schema": "vle-obstruction-timeline/v1",
        "tracked_families": [["n0", "n1", "n2"]],
        "snapshots": [
            {"version": "v1", "effective_date": "2026-01-01", "request": req},
            {"version": "v2", "effective_date": "2026-02-01", "request": {**req, "cells": req["cells"][:2]}},
        ],
    }
    r = obstruction(timeline)
    assert r["intervals"][0]["birth"] == "v1" and r["intervals"][0]["death"] == "v2"
    assert r["intervals"][0]["left_censored"]
    assert obstruction({**timeline, "max_states": 1})["status"] == "unknown"
