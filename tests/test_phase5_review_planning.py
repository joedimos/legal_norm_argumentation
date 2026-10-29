import json
from pathlib import Path

import jsonschema

from src.review_planning import plan_review
from src.review_simulation import simulate

CORPUS = json.loads(Path("benchmarks/review-planning-v1.json").read_text())


def test_minimax_questions_are_suggestions_and_costs_are_exact_in_scope():
    r = plan_review(CORPUS["cases"][0]["request"])
    jsonschema.validate(r, json.loads(Path("schemas/review-plan.schema.json").read_text()))
    assert r["status"] == "planned" and r["worst_case_cost"] == 1
    assert all(not p["auto_answered"] for p in r["review_prompts"])
    assert r == plan_review(CORPUS["cases"][0]["request"])
    assert plan_review({**CORPUS["cases"][0]["request"], "max_states": 1})["status"] == "unknown"


def test_opposite_verdict_witnesses_and_multiple_answer_domains():
    r = plan_review(CORPUS["cases"][3]["request"])
    assert r["status"] == "unresolvable" and len(r["opposite_verdict_witnesses"]) == 2
    assert len(r["certificates"]) == 2
    assert plan_review(CORPUS["cases"][5]["request"])["worst_case_cost"] == 1
    assert plan_review(CORPUS["cases"][4]["request"])["status"] == "already_decided"


def test_simulation_keeps_distributions_ties_and_insufficiency():
    r = simulate(CORPUS)
    assert len(r["rows"]) == 8 and r["real_reviewers"] == 0
    assert r["rows"][3]["status"] == "unresolvable"
    assert max(v["questions"] for v in r["rows"][0]["distributions"]["fixed"]) > max(
        v["questions"] for v in r["rows"][0]["distributions"]["minimax"]
    )
    assert len(r["rows"][0]["distributions"]["random"]) == 8
