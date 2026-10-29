import ast
import json
import random
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from src.audit_bundle import create_bundle, verify_bundle
from src.certificates import certify
from src.checker import check_certificate, evaluate, sha
from src.temporal import Formula, atom, combine, evaluate_formula


def test_independent_checker_has_no_reasoner_imports_and_stays_small():
    files = list(Path("src/checker").glob("*.py"))
    assert sum(len(p.read_text().splitlines()) for p in files) <= 500
    for p in files:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0 and node.module in {"__future__", "itertools", "time"}
            if isinstance(node, ast.Import):
                assert all(n.name in {"hashlib", "json"} for n in node.names)


def test_all_temporal_fixtures_and_seeded_random_formulas_differential():
    for c in json.loads(Path("benchmarks/research-v1.json").read_text())["cases"]:
        if c["kind"] == "temporal":
            f = c["input"]["formula"]
            trace = c["input"]["trace"]
            assert evaluate(f, trace) == evaluate_formula(Formula.from_dict(f), trace)
    rng = random.Random(17)

    def generate(depth):
        if not depth or rng.random() < 0.3:
            return atom(rng.choice(["p", "q"]))
        op = rng.choice(["not", "and", "or", "next", "weak_next", "eventually", "always", "until"])
        return combine(op, *[generate(depth - 1) for _ in range(2 if op in {"and", "or", "until"} else 1)])

    for _ in range(128):
        f = generate(3)
        for h in (1, 2, 3):
            trace = [{n: rng.choice([False, True, None]) for n in ["p", "q"]} for _ in range(h)]
            assert evaluate(f.to_dict(), trace) == evaluate_formula(f, trace)


def test_nonvacuous_certificates_bounds_and_corruption_rejection():
    requests = [{"property": {"op": "atom", "name": "p"}, "horizon": 1}, {"property": {"op": "true"}, "horizon": 2}]
    for req in requests:
        r = certify(req)
        c = r["certificate"]
        jsonschema.validate(c, json.loads(Path("schemas/certificate.schema.json").read_text()))
        assert check_certificate(c, expected_query_sha256=c["query_sha256"])["valid"]
        archive, root = create_bundle({"certificate.json": c})
        assert verify_bundle(archive, expected_manifest_digest=root)["valid"]
        bad = deepcopy(c)
        bad["evidence"]["unexpected"] = True
        assert not check_certificate(bad)["valid"]
        invalid, _ = create_bundle({"certificate.json": bad})
        assert not verify_bundle(invalid)["valid"]
    assert certify({"property": {"op": "atom", "name": "p"}, "horizon": 20, "max_states": 2})["status"] == "unknown"
    assert certify({"property": {"op": "true"}, "horizon": 1, "assumptions": [{"op": "false"}]})["status"] == "unknown"
    assert verify_bundle(create_bundle({"old.json": {"status": "unknown"}})[0])["valid"]


def test_mutants_unknown_horizon_priority_binding_and_escape_accounting():
    c = certify({"property": {"op": "atom", "name": "p"}, "horizon": 1})["certificate"]
    bad = deepcopy(c)
    bad["evidence"]["trace"][0]["p"] = None
    assert not check_certificate(bad)["valid"]
    compliance = certify({"property": {"op": "true"}, "horizon": 2})["certificate"]
    shorter = deepcopy(compliance)
    shorter["query"]["horizon"] = 1
    shorter["query_sha256"] = sha(shorter["query"])
    assert not check_certificate(shorter)["valid"]
    wrong = certify({"property": {"op": "true"}, "horizon": 1, "binding": {"priority": "reversed"}})["certificate"]
    assert check_certificate(wrong)["valid"]  # algebra alone cannot establish intended priority
    intended = {**wrong["query"], "binding": {"priority": "original"}}
    assert not check_certificate(wrong, expected_query_sha256=sha(intended))["valid"]
    bad = deepcopy(compliance)
    bad["evidence"]["merkle_root"] = "0" * 64
    assert not check_certificate(bad)["valid"]
    bad = deepcopy(compliance)
    bad["decision_authorized"] = True
    assert not check_certificate(bad)["valid"]


def test_observations_complete_witnesses_and_query_validation():
    r = certify({"property": {"op": "atom", "name": "p"}, "horizon": 1, "observations": [{"p": False}]})
    assert r["status"] == "violation"
    with pytest.raises(ValueError):
        certify({"property": {"op": "atom", "name": "p"}, "horizon": True})
    with pytest.raises(ValueError):
        evaluate({"op": "bad"}, [{}])
