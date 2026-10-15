from src.norm_extraction import analyze_norm_conflicts, extract_norms


def test_prohibition_and_compatibility_keys():
    text = "The data controller shall not process personal data without consent."
    norms = extract_norms(text, source_id="gdpr-example")
    assert len(norms) == 1
    norm = norms[0]
    assert norm["modality"] == "prohibition"
    assert norm["type"] == "prohibition"
    assert norm["sentence"] == text
    assert text[norm["source_span"][0] : norm["source_span"][1]] == text


def test_condition_and_exception_are_preserved():
    text = "If a breach is likely, the controller must notify the authority unless an exception applies."
    norm = extract_norms(text)[0]
    assert norm["condition"].lower().startswith("if a breach is likely")
    assert norm["actor"].lower() == "the controller"
    assert norm["action"].lower() == "notify the authority"
    assert norm["exception"].lower().startswith("unless an exception applies")


def test_reference_and_ambiguity_metadata():
    text = "Under Article 7, it must retain the record."
    norm = extract_norms(text)[0]
    assert any("Article 7".lower() == ref.lower() for ref in norm["references"])
    assert "possible_anaphora" in norm["ambiguity_flags"]


def test_norm_ids_stable_for_same_source_span():
    text = "The operator must retain logs."
    first = extract_norms(text, source_id="x")[0]["id"]
    second = extract_norms(text, source_id="x")[0]["id"]
    assert first == second


def test_simple_obligation_prohibition_conflict_detection():
    norms = [
        {"id": "n1", "modality": "obligation", "actor": "operator", "action": "retain audit logs"},
        {"id": "n2", "modality": "prohibition", "actor": "operator", "action": "retain the audit logs"},
    ]
    conflicts = analyze_norm_conflicts(norms)
    assert conflicts and conflicts[0]["requires_legal_priority_analysis"] is True
