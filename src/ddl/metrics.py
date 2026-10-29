"""Diagnostic vector; unreviewed expected fixtures are not expert gold standards."""

from __future__ import annotations

from ..verification import audit_extraction
from .model import parse_document


def metric_vector(predicted, expected, *, source_text="", extracted_candidates=()):
    gold, _ = parse_document(expected)
    try:
        pred, _ = parse_document(predicted)
        syntax = True
    except (ValueError, TypeError, KeyError):
        pred, syntax = [], False
    by_id = {r.id: r for r in pred}

    def ratio(count, total):
        return count / total if total else None

    def units(rules):
        result = set()
        for r in rules:
            result.add((r.id, "kind", r.kind))
            result.add((r.id, "consequent", str(r.consequent.to_list())))
            result.update((r.id, "condition", p) for p in r.body)
            result.update((r.id, "exception", p) for p in r.exceptions)
            result.update((r.id, "compensation:" + str(i), str(p.to_list())) for i, p in enumerate(r.compensation))
        return result

    required, recovered = units(gold), units(pred)
    required.update(("superiority", a, b) for a, b in expected["superiority"])
    if syntax:
        recovered.update(("superiority", a, b) for a, b in predicted["superiority"])
    exceptions = {(r.id, e) for r in gold for e in r.exceptions}
    matched_exceptions = {(r.id, e) for r in pred for e in r.exceptions} & exceptions
    audit = audit_extraction(source_text, list(extracted_candidates)) if source_text else None
    return {
        "syntactic_validity": syntax,
        "modality_accuracy": ratio(
            sum(r.id in by_id and by_id[r.id].consequent.modality == r.consequent.modality for r in gold), len(gold)
        ),
        "negation_placement": ratio(
            sum(
                r.id in by_id
                and (by_id[r.id].consequent.negated, by_id[r.id].consequent.atom.startswith("~"))
                == (r.consequent.negated, r.consequent.atom.startswith("~"))
                for r in gold
            ),
            len(gold),
        ),
        "exception_coverage": ratio(len(matched_exceptions), len(exceptions)),
        "normative_completeness": ratio(len(required & recovered), len(required)),
        "missing_normative_units": [list(x) for x in sorted(required - recovered)],
        "implicit_obligations": "requires_explicit_expected_units_human_review",
        "textual_fidelity": None
        if audit is None
        else {"candidate_findings": audit["candidate_findings"], "possible_omissions": audit["possible_omissions"]},
        "aggregate_score": None,
        "expert_accuracy_established": False,
    }
