import json
from pathlib import Path

from src.geometry import geometry
from src.geometry_study import run_study
from src.persistence import rips
from src.phase4 import Budget


def request():
    return {
        "schema": "vle-geometry/v1",
        "horizon": 1,
        "topology": True,
        "interpretations": [
            {"id": "p", "property": {"op": "atom", "name": "p"}},
            {"id": "q", "property": {"op": "atom", "name": "q"}},
            {"id": "p_again", "property": {"op": "atom", "name": "p"}},
            {"id": "not_p", "property": {"op": "not", "args": [{"op": "atom", "name": "p"}]}},
        ],
    }


def test_exact_metric_axioms_on_behavior_quotient_and_determinism():
    r = geometry(request())
    counts = r["disagreement_counts"]
    assert r["status"] == "reported" and r == geometry(request())
    for i in range(4):
        assert counts[i][i] == 0
        for j in range(4):
            assert counts[i][j] == counts[j][i]
            for k in range(4):
                assert counts[i][k] <= counts[i][j] + counts[j][k]
    assert ["p", "p_again"] in r["equivalence_classes"] and r["selected_id"] is None
    assert r["persistence"]["largest_finite_H0_lifetime"] >= 0


def test_shared_sample_intervals_and_bounds():
    r = geometry({**request(), "samples": 128, "seed": 23})
    assert r == geometry({**request(), "samples": 128, "seed": 23})
    assert r["confidence"] == 0.95 and r["triangle_violations"] == []
    exact = geometry(request())["distances"]
    assert all(r["intervals"][i][j][0] <= exact[i][j] <= r["intervals"][i][j][1] for i in range(4) for j in range(4))
    assert geometry({**request(), "max_states": 1})["status"] == "unknown"


def test_triangle_faces_kill_rips_h1_square_loop():
    matrix = [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
    r = rips(matrix, 2, Budget())
    assert any(b["dimension"] == 1 and b["birth"] == 0.5 and b["death"] == 1 for b in r["bars"])


def test_predeclared_synthetic_annotation_pipeline_executes_without_expert_claims():
    corpus = json.loads(Path("benchmarks/geometry-study-v1.json").read_text())
    r = run_study(corpus)
    assert len(r["rows"]) == 8 and r["real_reviewers"] == 0 and r["simulated_only"]
    assert r["real_predictive_validity"] == "unmeasured" and not r["decision_authorized"]
    assert r == run_study(corpus)
