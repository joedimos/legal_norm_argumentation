from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, Optional


SEMANTIC_FIELDS = (
    "source_id",
    "source_span",
    "sentence",
    "modality",
    "actor",
    "action",
    "condition",
    "exception",
    "references",
    "ambiguity_flags",
)


def semantic_fingerprint(norm: Dict) -> str:
    payload = {field: norm.get(field) for field in SEMANTIC_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewRecord:
    norm_id: str
    decision: str
    reviewer: str
    fingerprint: str
    reviewed_at: str
    note: Optional[str] = None


class ReviewLedger:
    """Human review ledger that invalidates approval after semantic edits."""

    VALID_DECISIONS = {"approved", "rejected", "needs_revision"}

    def __init__(self) -> None:
        self._records: Dict[str, ReviewRecord] = {}

    def record(self, norm: Dict, decision: str, reviewer: str, note: Optional[str] = None) -> ReviewRecord:
        if decision not in self.VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(self.VALID_DECISIONS)}")
        norm_id = norm.get("id")
        if not norm_id:
            raise ValueError("norm must have an id")
        if not reviewer.strip():
            raise ValueError("reviewer must be non-empty")
        record = ReviewRecord(
            norm_id=norm_id,
            decision=decision,
            reviewer=reviewer,
            fingerprint=semantic_fingerprint(norm),
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            note=note,
        )
        self._records[norm_id] = record
        return record

    def status(self, norm: Dict) -> str:
        record = self._records.get(norm.get("id"))
        if record is None:
            return "unreviewed"
        if record.fingerprint != semantic_fingerprint(norm):
            return "stale"
        return record.decision

    def get(self, norm_id: str) -> Optional[ReviewRecord]:
        return self._records.get(norm_id)


def assumption_review_prompts(corrections):
    """Review queue payloads only: no ledger record, reviewer or approval is created."""
    return [{'kind': 'assumption_relaxation', 'review_status': 'unreviewed',
             'assumption_ids': c['assumption_ids'], 'source_spans': [a['source_span'] for a in c['assumptions']],
             'verdict_change': c['verdict_change'], 'witness': c['witness'],
             'text': 'Review whether these assumptions may be relaxed: ' +
             '; '.join(a['description'] for a in c['assumptions']) +
             '. This only restores a feasible bounded completion, not universal compliance.'}
            for c in corrections if c['assumption_ids']]


def suggested_review_order(plan,questions):
    """Suggestions for existing human queues; never create decisions or answers."""
    indexed={q['id']:q for q in questions}
    return [{'node_id':n['id'],'question_id':n['question_id'],'kind':indexed[n['question_id']]['kind'],
             'prompt':indexed[n['question_id']]['prompt'],'review_status':'unreviewed','auto_answered':False}
            for n in plan['nodes'] if 'question_id' in n]
