import json
from pathlib import Path

from src.metamorphic import metamorphic, transform

CORPUS = json.loads(Path("benchmarks/metamorphic-v1.json").read_text())


def test_all_relations_run_and_negative_results_remain_visible():
    r = metamorphic(CORPUS)
    assert r["status"] == "reported" and len(r["by_relation"]) == 12
    assert r["metrics"]["violated"] > 0 and r["metrics"]["unknown"] > 0
    assert r == metamorphic(CORPUS)
    assert r["by_relation"]["double_negation"]["MVR"] == 1
    assert r["by_relation"]["weaken_modality"]["MVR"] == 1
    row = next(r for r in r["rows"] if r["relation"] == "weaken_modality")
    assert row["actual"] == "equivalent_bounded" and row["normative_output_relation"] == "equivalent"


def test_planted_fake_fault_detected_without_relabeling_inconclusive():
    corpus = {**CORPUS, "cases": [c for c in CORPUS["cases"] if c["id"] == "except-o"], "fault": "ignore_transform"}
    r = metamorphic(corpus, provider="fake")
    assert r["metrics"]["violated"] == 1 and r["metrics"]["MVR"] == 1
    assert r["rows"][0]["provenance"]["model_id"] == "deterministic-fake/v1"
    assert not r["decision_authorized"]


def test_compositions_bounded_and_mechanical_applicability():
    case = CORPUS["cases"][0]
    assert transform("Unrelated text.", "voice", case) == []
    assert metamorphic({**CORPUS, "max_states": 1})["status"] == "unknown"
    r = metamorphic(CORPUS)
    assert all(not v["infinite_robustness_claimed"] for v in r["robustness_radius"])
