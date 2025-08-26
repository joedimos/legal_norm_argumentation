from src.norm_extraction import extract_norms
def test_rule_extractor_simple():
    text = "The data controller shall not process personal data without consent."
    norms = extract_norms(text, use_transformer=False)
    assert any(n["modality"] == "prohibition" or "shall not" in n["sentence"].lower() for n in norms)
