"""Canonical local DDL/1 text. External dialects require an explicit adapter."""

from __future__ import annotations

import json

from ..normative_model import InterpretationSet, Modality
from .model import ARROWS, DDLRule, ModalLiteral, document, parse_document


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def export_document(raw):
    rules, superiority = parse_document(raw)
    lines = ["DDL/1"]
    for r in sorted(rules, key=lambda r: r.id):
        modal = ("~" if r.consequent.negated else "") + r.consequent.code
        lines.append(
            f"{r.id}: {compact(list(r.body))} {ARROWS[r.kind]} {modal}({compact(r.consequent.atom)}) "
            f"UNLESS {compact(list(r.exceptions))} COMPENSATE {compact([c.to_list() for c in r.compensation])}"
        )
    lines += [f"superiority {a} > {b}" for a, b in superiority]
    return {"text": "\n".join(lines) + "\n", "document": document(rules, superiority), "lossy_fields": []}


def export_norms(value):
    norms = value.interpretations if isinstance(value, InterpretationSet) else tuple(value)
    if len(norms) > 128:
        raise ValueError("at most 128 norms")
    rules, lost = [], []
    for n in norms:
        if n.modality == Modality.RECOMMENDATION:
            lost.append(f"{n.id}.modality:recommendation_not_in_DDL_fragment")
            continue
        # Stable local rule ids; retain original identifiers in metadata sidecar.
        from ..audit_bundle import digest

        id = "r_" + digest(n.id.encode())[:24]
        target = "action:" + compact([n.subject, n.action, n.object])
        rules.append(
            DDLRule(
                id,
                "defeasible",
                tuple(p.canonical() for p in n.conditions),
                ModalLiteral(n.modality, target),
                tuple(p.canonical() for p in n.exceptions),
            )
        )
        for key, present in [
            ("deadline", n.deadline),
            ("references", n.references),
            ("authority", n.authority != type(n.authority)()),
            ("assumptions", n.assumptions),
            ("confidence", n.confidence != 1),
            ("provenance", n.provenance),
        ]:
            if present:
                lost.append(f"{n.id}.{key}:preserved_only_in_sidecar_not_DDL_semantics")
    result = export_document(document(rules))
    result.update(
        lossy_fields=sorted(lost),
        metadata_sidecar={
            "norms": [n.to_dict() for n in norms],
            "interpretation_set": {
                "source_norm_id": value.source_norm_id,
                "selected_id": value.selected_id,
                "selection_rationale": value.selection_rationale,
            }
            if isinstance(value, InterpretationSet)
            else None,
        },
    )
    return result
