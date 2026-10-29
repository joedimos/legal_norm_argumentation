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
    assert r["by_relation"]["weaken_modality"]["MVR"] == 0
    row = next(r for r in r["rows"] if r["relation"] == "weaken_modality")
    assert row["actual"] == "strict_weakening" and row["normative_output_relation"] == "incomparable"


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


def test_preserved_baseline_and_token_boundary_treatment_are_distinct():
    from src.norm_extraction import classify_norm_type

    baseline = json.loads(Path("benchmarks/phase5-metamorphic-results.json").read_text())
    assert baseline["extractor"]["by_relation"]["weaken_modality"]["MVR"] == 1
    assert classify_norm_type("The operator must notify.") == "obligation"
    assert classify_norm_type("The operator may notify.") == "permission"
    assert classify_norm_type("The operator must not notify.") == "prohibition"
    assert classify_norm_type("The operator has a shoulder injury.") == "unknown"


def test_live_adapter_is_exercised_only_with_fake_transport_and_full_provenance():
    from src.metamorphic import extract_one
    from src.redundancy.generate import RuntimeProvider

    calls = []

    def transport(payload, **kwargs):
        prompt = json.loads(payload["messages"][1]["content"])
        calls.append(prompt["source"])
        return {
            "done": True,
            "model": "fake",
            "message": {"content": json.dumps(extract_one(prompt["source"]).to_dict())},
        }

    runtime = RuntimeProvider({"model": "fake", "provider": "vllm"}, transport=transport)
    corpus = {**CORPUS, "cases": [c for c in CORPUS["cases"] if c["id"] == "modality-en"]}
    result = metamorphic(corpus, provider="vllm", runtime=runtime)
    assert result["status"] == "reported" and len(calls) == 2
    assert result["rows"][0]["provenance"]["generation_request"]["model"] == "fake"
    assert result["base_formalizations"][0]["review_status"] == "unreviewed"
