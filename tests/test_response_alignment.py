from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.audit_bundle import digest, verify_bundle
from src.family_contract import record_family_review
from src.interpretation_family import analyze_family
from src.response_alignment import bind_candidate, run_revision_sequence
from src.temporal import atom, combine

DISCLOSE = atom("response:disclose")


def snapshot(id, kind, text):
    return {"id": id, "kind": kind, "text": text, "sha256": digest(text.encode()), "version": "1", "language": "en",
            **({"jurisdiction": "FR", "effective_from": None, "effective_to": None} if kind == "normative" else {})}


def ref(doc):
    return {"document_id": doc["id"], "start": 0, "end": len(doc["text"]), "quote": doc["text"]}


def family():
    doc = snapshot("policy", "normative", "Synthetic policy: do not disclose confidential records.")
    payload = {"schema": "vle-family/v1", "context": {"jurisdiction": "FR", "as_of": "2026-10-27", "actor": "fixture",
               "capacity": "research"}, "documents": [doc], "horizon": 1,
               "interpretations": [{"id": "i", "property": combine("not", DISCLOSE).to_dict(), "source_refs": [ref(doc)],
                                    "rationale": "Synthetic test interpretation, not an expert legal annotation."}]}
    return record_family_review(payload, scope="family", reviewer="fixture only", notes="Self-reported test receipt.")


def candidate(base, id, discloses, *, reviewed=True):
    doc = snapshot(id, "draft", "I will disclose the records." if discloses else "I will withhold the records.")
    behavior = {"document_id": id, "action_atoms": ["response:disclose"], "mappings": [{"id": "literal-intent",
                "constraints": [(DISCLOSE if discloses else combine("not", DISCLOSE)).to_dict()],
                "source_refs": [ref(doc)], "rationale": "Synthetic intended-action mapping, not an observation."}]}
    result = {"document": doc, "behavior": behavior}
    if reviewed:
        bound = record_family_review(bind_candidate(base, result), scope="behavior", reviewer="fixture mapper",
                                     notes="Self-reported mapping approval for regression test only.")
        result["reviews"] = {"behavior": bound["reviews"]["behavior"]}
    return result


def test_bounded_sequence_rechecks_and_revises_without_promoting_claims_to_facts():
    base = family()
    first, second = candidate(base, "first", True), candidate(base, "second", False)
    payload = {"schema": "vle-revision/v1", "family": base, "candidates": [first, second]}
    result = run_revision_sequence(payload)
    assert result["status"] == "model_satisfied"
    assert [s["feedback"]["status"] for s in result["steps"]] == ["robust_violated", "robust_satisfied"]
    assert result["steps"][0]["feedback"]["counterexample"]["trace"][0]["response:disclose"]
    assert result["selected_candidate"] == "second"
    assert result["generation_calls"] == 0
    assert not result["deployment_authorized"]
    assert "observations" not in bind_candidate(base, first)
    assert "behavior" not in base


def test_missing_mapping_approval_stops_before_supplied_revision():
    base = family()
    result = run_revision_sequence({"schema": "vle-revision/v1", "family": base,
                                    "candidates": [candidate(base, "first", True, reviewed=False), candidate(base, "second", False)]})
    assert result["status"] == "review_required"
    assert len(result["steps"]) == 1
    assert result["selected_candidate"] is None


def test_revised_text_or_constraints_invalidate_mapping_approval():
    base = family()
    item = candidate(base, "first", True)
    item["behavior"]["mappings"][0]["constraints"] = [combine("not", DISCLOSE).to_dict()]
    result = analyze_family(bind_candidate(base, item))
    assert result["analysis"]["status"] == "robust_satisfied"
    assert result["status"] == "review_required"


def test_behavior_alternatives_cannot_be_queried_as_external_facts():
    base = family()
    item = candidate(base, "unclear", True)
    mapping = deepcopy(item["behavior"]["mappings"][0])
    mapping["id"] = "other-reading"
    mapping["constraints"] = [combine("not", DISCLOSE).to_dict()]
    item["behavior"]["mappings"].append(mapping)
    bound = bind_candidate(base, item)
    result = analyze_family(bound)
    assert result["analysis"]["status"] == "grounding_disagreement"
    assert result["scope"]["queries"] == []
    witnesses = result["analysis"]["indistinguishable_witnesses"]
    assert witnesses[0]["interpretation_id"] == witnesses[1]["interpretation_id"]
    assert witnesses[0]["behavior_id"] != witnesses[1]["behavior_id"]
    bound["queries"] = [{"step": 0, "atom": "response:disclose", "cost": 1}]
    with pytest.raises(ValueError, match="not factual queries"):
        analyze_family(bound)


def test_behavior_contract_cannot_assume_authorization_or_trust_draft_action_as_observed():
    base = family()
    item = candidate(base, "first", True)
    item["behavior"]["mappings"][0]["constraints"] = [atom("fact:authorized").to_dict()]
    with pytest.raises(ValueError, match="external facts"):
        analyze_family(bind_candidate(base, item))
    item = candidate(base, "first", True)
    bound = bind_candidate(base, item)
    bound["observations"] = [{"response:disclose": True}]
    with pytest.raises(ValueError, match="cannot be trusted observations"):
        analyze_family(bound)


def test_cycle_cap_and_stall_do_not_release_a_candidate():
    base = family()
    first, second = candidate(base, "first", True), candidate(base, "second", False)
    result = run_revision_sequence({"schema": "vle-revision/v1", "family": base, "candidates": [first, second], "max_cycles": 1})
    assert result["status"] == "revision_limit"
    assert result["selected_candidate"] is None
    assert run_revision_sequence({"schema": "vle-revision/v1", "family": base, "candidates": [first, first]})["status"] == "stalled"


def test_candidate_cannot_override_normative_review():
    base = family()
    item = candidate(base, "first", True)
    item["reviews"]["family"] = base["reviews"]["family"]
    with pytest.raises(ValueError, match="may not overwrite"):
        bind_candidate(base, item)


def test_cli_sequence_audit_retains_candidates_and_findings(tmp_path):
    base = family()
    source, output = tmp_path / "sequence.json", tmp_path / "sequence.zip"
    source.write_text(json.dumps({"schema": "vle-revision/v1", "family": base,
                                  "candidates": [candidate(base, "first", True), candidate(base, "second", False)]}))
    result = subprocess.run([sys.executable, "-m", "src.cli", "statutory-revise", "--input", str(source), "--bundle", str(output)],
                            cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "model_satisfied"
    assert verify_bundle(output.read_bytes(), expected_manifest_digest=payload["bundle_manifest_digest"])["valid"]


def test_underspecified_behavior_mapping_stays_unknown_to_factual_queries():
    from src.temporal import Formula
    base = family()
    item = candidate(base, "vague", True, reviewed=False)
    item["behavior"]["mappings"][0]["constraints"] = [Formula("true").to_dict()]
    bound = record_family_review(bind_candidate(base, item), scope="behavior", reviewer="fixture",
                                 notes="Receipt does not remove underspecified behavior.")
    result = analyze_family(bound)
    assert result["status"] == "behavior_uncertainty"
    assert result["analysis"]["plan"] is None


def test_infeasible_candidate_mapping_is_not_dropped_from_family():
    base = family()
    item = candidate(base, "inconsistent", True, reviewed=False)
    item["behavior"]["mappings"][0]["constraints"] = [DISCLOSE.to_dict(), combine("not", DISCLOSE).to_dict()]
    result = analyze_family(bind_candidate(base, item))
    assert result["analysis"]["status"] == "inconsistent_family"
    assert result["analysis"]["infeasible_models"]
    assert result["analysis"]["plan"] is None
