from __future__ import annotations

import re
from typing import Dict, Iterable, List


_DEONTIC_HINT = re.compile(
    r"\b(shall(?:\s+not)?|must(?:\s+not)?|may(?:\s+not)?|should|ought\s+to|"
    r"required\s+to|prohibited\s+from|permitted\s+to|allowed\s+to)\b",
    re.IGNORECASE,
)
_CONDITION_HINT = re.compile(r"\b(if|when|whenever|where|provided\s+that|subject\s+to)\b", re.IGNORECASE)
_EXCEPTION_HINT = re.compile(r"\b(unless|except|save\s+where|provided\s+however)\b", re.IGNORECASE)


def _sentence_spans(text: str) -> Iterable[tuple[str, int, int]]:
    for match in re.finditer(r"\S(?:.*?\S)?(?:[.!?](?=\s|$)|$)", text, re.DOTALL):
        raw = match.group(0)
        for part in re.finditer(r"[^\n]+", raw):
            sentence = part.group(0).strip()
            if not sentence:
                continue
            left = len(part.group(0)) - len(part.group(0).lstrip())
            start = match.start() + part.start() + left
            yield sentence, start, start + len(sentence)


def verify_candidate(text: str, norm: Dict) -> List[Dict]:
    findings: List[Dict] = []
    span = norm.get("source_span")
    if not isinstance(span, list) or len(span) != 2 or not all(isinstance(x, int) for x in span):
        findings.append({"code": "invalid_source_span", "severity": "error"})
        source = norm.get("sentence", "")
    else:
        start, end = span
        source = text[start:end] if 0 <= start <= end <= len(text) else ""
        if not source or source != norm.get("sentence"):
            findings.append({"code": "source_span_mismatch", "severity": "error"})

    if not norm.get("actor"):
        findings.append({"code": "missing_actor", "severity": "warning"})
    if not norm.get("action"):
        findings.append({"code": "missing_action", "severity": "error"})
    if norm.get("modality") in (None, "unknown"):
        findings.append({"code": "unknown_modality", "severity": "warning"})
    if _CONDITION_HINT.search(source) and not norm.get("condition"):
        findings.append({"code": "possible_condition_omission", "severity": "warning"})
    if _EXCEPTION_HINT.search(source) and not norm.get("exception"):
        findings.append({"code": "possible_exception_omission", "severity": "warning"})

    return findings


def audit_extraction(text: str, norms: List[Dict]) -> Dict:
    findings = {norm.get("id", f"candidate-{i}"): verify_candidate(text, norm) for i, norm in enumerate(norms)}

    covered_spans = {
        tuple(norm.get("source_span", []))
        for norm in norms
        if isinstance(norm.get("source_span"), list) and len(norm["source_span"]) == 2
    }
    omissions: List[Dict] = []
    for sentence, start, end in _sentence_spans(text):
        if _DEONTIC_HINT.search(sentence) and (start, end) not in covered_spans:
            omissions.append(
                {
                    "source_span": [start, end],
                    "sentence": sentence,
                    "code": "possible_norm_omission",
                }
            )

    error_count = sum(item["severity"] == "error" for values in findings.values() for item in values)
    warning_count = sum(item["severity"] == "warning" for values in findings.values() for item in values)
    denominator = max(1, len(norms) * 2 + len(omissions))
    score = max(0.0, 1.0 - (2 * error_count + warning_count + 2 * len(omissions)) / denominator)

    return {
        "candidate_findings": findings,
        "possible_omissions": omissions,
        "summary": {
            "candidate_count": len(norms),
            "error_count": error_count,
            "warning_count": warning_count,
            "possible_omission_count": len(omissions),
            "faithfulness_diagnostic_score": round(score, 4),
            "certified": False,
        },
    }
