from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Sequence

from .certification import readiness_report
from .compliance import evaluate
from .formalization import formalize_candidates, select_interpretation
from .normative_model import FactContext, InterpretationSet, SourceAuthority
from .traceability import EvidenceGraph


def _document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_norms(sets: Sequence[InterpretationSet]):
    result = []
    unresolved = []
    for item in sets:
        selected = item.selected()
        if selected is not None:
            result.append(selected)
        else:
            unresolved.append(item.source_norm_id)
    return result, unresolved


def run_lifecycle(
    *,
    text: str,
    source_id: str,
    candidates: Sequence[Mapping],
    extraction_audit: Mapping,
    context: FactContext,
    authorities: Optional[Mapping[str, SourceAuthority]] = None,
    review_statuses: Optional[Sequence[str]] = None,
    interpretation_selections: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict:
    formal_sets = formalize_candidates(candidates, authorities=authorities)
    selections = interpretation_selections or {}
    resolved_sets = []
    for item in formal_sets:
        selection = selections.get(item.source_norm_id)
        if selection:
            selected_id = selection.get("selected_id")
            rationale = selection.get("rationale")
            if selected_id and rationale:
                item = select_interpretation(item, selected_id, rationale)
        resolved_sets.append(item)
    formal_sets = resolved_sets

    selected, unresolved = selected_norms(formal_sets)
    compliance = evaluate(selected, context) if selected else {
        "status": "indeterminate",
        "findings": [],
        "violations": [],
        "reasoning": {"unresolved_conflicts": []},
    }

    evidence = EvidenceGraph()
    source_node = f"source:{source_id}:{_document_hash(text)[:12]}"
    evidence.add(source_node, "source_document", {"source_id": source_id, "hash": _document_hash(text)})
    for candidate in candidates:
        cid = f"candidate:{candidate['id']}"
        evidence.add(cid, "norm_candidate", candidate)
        evidence.link(source_node, cid, "extracted_from")
    for item in formal_sets:
        for interpretation in item.interpretations:
            fid = f"formal:{interpretation.id}"
            evidence.add(
                fid,
                "formal_interpretation",
                interpretation.to_dict(),
                selected=item.selected_id == interpretation.id,
                selection_rationale=item.selection_rationale,
            )
            evidence.link(f"candidate:{item.source_norm_id}", fid, "formalizes")
    decision_id = f"compliance:{evidence._hash(compliance)[:16]}"
    evidence.add(decision_id, "compliance_report", compliance)
    for norm in selected:
        evidence.link(f"formal:{norm.id}", decision_id, "contributes_to")

    readiness = readiness_report(
        extraction_audit=extraction_audit,
        review_statuses=list(review_statuses or []),
        unresolved_interpretations=len(unresolved),
        unresolved_conflicts=len(compliance.get("reasoning", {}).get("unresolved_conflicts", [])),
    )
    readiness_id = f"readiness:{evidence._hash(readiness)[:16]}"
    evidence.add(readiness_id, "readiness_report", readiness)
    evidence.link(decision_id, readiness_id, "evidence_for")

    return {
        "formalizations": [
            {
                "source_norm_id": item.source_norm_id,
                "selected_id": item.selected_id,
                "selection_rationale": item.selection_rationale,
                "unresolved": item.unresolved,
                "interpretations": [x.to_dict() for x in item.interpretations],
            }
            for item in formal_sets
        ],
        "compliance": compliance,
        "readiness": readiness,
        "evidence": evidence.export(),
    }
