from copy import deepcopy

from src.norm_extraction import extract_norms
from src.review import ReviewLedger
from src.verification import audit_extraction, verify_candidate


def test_review_becomes_stale_after_semantic_change():
    norm = extract_norms("The operator must retain logs.", source_id="p1")[0]
    ledger = ReviewLedger()
    ledger.record(norm, "approved", reviewer="reviewer-1")
    assert ledger.status(norm) == "approved"

    edited = deepcopy(norm)
    edited["action"] = "delete logs"
    assert ledger.status(edited) == "stale"


def test_nonsemantic_confidence_change_does_not_invalidate_review():
    norm = extract_norms("The operator must retain logs.", source_id="p1")[0]
    ledger = ReviewLedger()
    ledger.record(norm, "approved", reviewer="reviewer-1")
    edited = deepcopy(norm)
    edited["confidence"] = 0.99
    assert ledger.status(edited) == "approved"


def test_verifier_detects_tampered_source_span():
    text = "The operator must retain logs."
    norm = extract_norms(text)[0]
    norm["source_span"] = [0, 5]
    codes = {finding["code"] for finding in verify_candidate(text, norm)}
    assert "source_span_mismatch" in codes


def test_audit_detects_omitted_deontic_sentence():
    text = "The operator must retain logs. The auditor may inspect them."
    norms = extract_norms(text)
    report = audit_extraction(text, norms[:1])
    assert report["possible_omissions"]
    assert report["summary"]["possible_omission_count"] == 1
    assert report["summary"]["certified"] is False


def test_audit_has_no_omission_for_full_extraction():
    text = "If a breach occurs, the operator must notify the authority unless reporting is legally barred."
    norms = extract_norms(text)
    report = audit_extraction(text, norms)
    assert report["possible_omissions"] == []
