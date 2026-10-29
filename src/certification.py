from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    explanation: str


def readiness_report(
    *,
    extraction_audit: Mapping,
    review_statuses: Sequence[str],
    unresolved_interpretations: int,
    unresolved_conflicts: int,
    benchmark_metrics: Mapping | None = None,
    phase4_evidence: Mapping | None = None,
    phase4_review_ledger=None,
    required_extraction_f1: float = 0.9,
    required_compliance_accuracy: float = 0.9,
) -> dict:
    """Evidence-gated research readiness; never claims legal certification."""
    summary = extraction_audit.get("summary", {})
    gates = [
        GateResult("source_faithfulness", summary.get("error_count", 1) == 0 and summary.get("possible_omission_count", 1) == 0, "No extraction errors or detected omissions."),
        GateResult("human_review", bool(review_statuses) and all(status == "approved" for status in review_statuses), "Every normative representation has current human approval."),
        GateResult("interpretation_resolution", unresolved_interpretations == 0, "All material interpretation alternatives have an explicit selection."),
        GateResult("conflict_resolution", unresolved_conflicts == 0, "No applicable normative conflict remains unresolved."),
    ]
    if benchmark_metrics is not None:
        extraction_f1 = benchmark_metrics.get("extraction", {}).get("detection", {}).get("f1", 0.0)
        compliance_accuracy = benchmark_metrics.get("compliance", {}).get("accuracy", 0.0)
        gates.extend([
            GateResult("benchmark_extraction", extraction_f1 >= required_extraction_f1, f"Extraction F1 {extraction_f1:.3f} >= {required_extraction_f1:.3f}."),
            GateResult("benchmark_compliance", compliance_accuracy >= required_compliance_accuracy, f"Compliance accuracy {compliance_accuracy:.3f} >= {required_compliance_accuracy:.3f}."),
        ])
    passed = all(gate.passed for gate in gates)
    return {
        "status": "research_ready" if passed else "not_ready",
        "legal_certification": False,
        "gates": [asdict(gate) for gate in gates],
        **({"advisory_gates": phase4_advisory_gates(phase4_evidence, review_ledger=phase4_review_ledger)}
           if phase4_evidence is not None else {}),
        "failed_gates": [gate.gate for gate in gates if not gate.passed],
        "disclaimer": "This report is an evidence gate for a research workflow, not a legal, regulatory, or standards certification.",
    }


def assumption_review_subject(report):
    """Bind an existing ReviewLedger decision to the exact assumption report."""
    from .audit_bundle import canonical_json, digest

    raw = canonical_json(report)
    return {'id': 'phase4-assumptions-' + digest(raw), 'source_id': 'phase4-assumptions',
            'sentence': raw.decode(), 'ambiguity_flags': ['assumption-review']}


def phase4_advisory_gates(evidence, *, review_ledger=None):
    """Advisory evidence only. Does not alter the pre-existing readiness status."""
    redundancy = evidence.get('redundancy', [])
    coverage = (isinstance(redundancy, list) and bool(redundancy) and all(
        r.get('schema') == 'vle-redundancy-report/v1' and r.get('decision_authorized') is False
        and r.get('status') != 'unknown' and len(r.get('candidates', [])) >= 2 for r in redundancy))
    faith = evidence.get('faithfulness', {})
    reported = (faith.get('schema') == 'vle-faithfulness-report/v1' and faith.get('status') == 'reported'
                and faith.get('evaluated_count', 0) > 0 and faith.get('evaluated_count') == faith.get('case_count')
                and {'direct_vs_symbolic', 'direct_vs_gold', 'symbolic_vs_gold'} <= set(faith.get('disagreement_rates', {})))
    assumptions = evidence.get('assumptions', {})
    reviewed = (assumptions.get('schema') == 'vle-assumption-report/v1' and assumptions.get('status') != 'unknown'
                and assumptions.get('truncated') is False and review_ledger is not None
                and review_ledger.status(assumption_review_subject(assumptions)) == 'approved')
    return [{'gate': gate, 'passed': passed, 'advisory': True, 'explanation': explanation}
            for gate, passed, explanation in [
                ('redundancy-coverage', coverage, 'Bounded comparisons supplied; agreement is not source faithfulness.'),
                ('faithfulness-gap-reported', reported, 'All three disagreements reported; no minimum accuracy asserted.'),
                ('assumption-review-completed', reviewed, 'Existing ledger approval must bind the complete report fingerprint.')]]
