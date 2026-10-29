from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

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
    authorities: Mapping[str, SourceAuthority] | None = None,
    review_statuses: Sequence[str] | None = None,
    interpretation_selections: Mapping[str, Mapping[str, str]] | None = None,
    reasoning_backend: str = "core",
    pyreason_timeout_seconds: int = 60,
) -> dict:
    if reasoning_backend not in {"core", "pyreason"}:
        raise ValueError("unsupported normative reasoning backend")
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
    if reasoning_backend == "pyreason":
        from .pyreason_normative import unknown_report

        if (not selected or unresolved or review_statuses is None or len(review_statuses) != len(candidates)
                or any(status != "approved" for status in review_statuses)):
            compliance = {"status": "indeterminate", "findings": [], "violations": [],
                          "reasoning": {"unresolved_conflicts": [], "backend": unknown_report(
                              "approved_unambiguous_formalizations_required", pyreason_timeout_seconds)}}
        else:
            compliance = evaluate(selected, context, backend="pyreason", pyreason_timeout_seconds=pyreason_timeout_seconds)
    else:
        compliance = evaluate(selected, context) if selected else {
            "status": "indeterminate", "findings": [], "violations": [],
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
    if reasoning_backend == "pyreason":
        backend_report = compliance["reasoning"]["backend"]
        backend_id = f"backend:pyreason:{evidence._hash(backend_report)[:16]}"
        evidence.add(backend_id, "normative_backend_computation", backend_report,
                     role="required_activation_backend", status=backend_report["status"])
        for norm in selected:
            evidence.link(f"formal:{norm.id}", backend_id, "compiled_for")
        evidence.link(backend_id, decision_id, "computes_activation_for")

    readiness = readiness_report(
        extraction_audit=extraction_audit,
        review_statuses=(list(review_statuses) if review_statuses is not None and len(review_statuses) == len(candidates)
                         else ["unreviewed"]),
        unresolved_interpretations=len(unresolved),
        unresolved_conflicts=len(compliance.get("reasoning", {}).get("unresolved_conflicts", [])),
    )
    if reasoning_backend == "pyreason":
        completed = compliance["reasoning"]["backend"]["status"] == "computed"
        readiness["gates"].append({"gate": "required_pyreason_computation", "passed": completed,
                                   "explanation": "The selected normative backend must complete and match its independent check."})
        if not completed:
            readiness["failed_gates"].append("required_pyreason_computation")
            readiness["status"] = "not_ready"
    if compliance["status"] == "indeterminate" or any(
        finding["status"] == "unknown" for finding in compliance.get("findings", [])
    ):
        readiness["gates"].append({"gate": "determinate_reasoning", "passed": False,
                                   "explanation": "Unknown applicability or unresolved argumentation blocks readiness."})
        readiness["failed_gates"].append("determinate_reasoning")
        readiness["status"] = "not_ready"
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
        **({"reasoning_backend": "pyreason", "decision_authorized": False} if reasoning_backend == "pyreason" else {}),
    }
