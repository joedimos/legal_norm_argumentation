from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence


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
    benchmark_metrics: Optional[Mapping] = None,
    required_extraction_f1: float = 0.9,
    required_compliance_accuracy: float = 0.9,
) -> Dict:
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
        "failed_gates": [gate.gate for gate in gates if not gate.passed],
        "disclaimer": "This report is an evidence gate for a research workflow, not a legal, regulatory, or standards certification.",
    }
