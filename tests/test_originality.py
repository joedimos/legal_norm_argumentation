from src.originality import LogicBasedLegalReasoner


def test_originality_case_uses_shared_aaf_and_resets_between_cases():
    reasoner = LogicBasedLegalReasoner()
    first = reasoner.analyze_legal_case(
        "The work reflects the author's personality and free creative choices, "
        "but some elements are dictated by technical function."
    )
    assert first["arguments"]
    assert first["attacks"]
    assert "grounded" in first["aaf"]

    second = reasoner.analyze_legal_case("No originality marker is present here.")
    assert second["arguments"] == {}
    assert second["attacks"] == []
