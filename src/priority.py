from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence

from .normative_model import NormInterpretation


@dataclass(frozen=True)
class PriorityDecision:
    winner_id: Optional[str]
    loser_id: Optional[str]
    principle: str
    status: str
    explanation: str


def actions_conflict(left: NormInterpretation, right: NormInterpretation) -> bool:
    if left.subject.lower() != right.subject.lower() or left.action.lower() != right.action.lower():
        return False
    pair = {left.modality.value, right.modality.value}
    return pair in ({"obligation", "prohibition"}, {"permission", "prohibition"})


def _date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def resolve_pair(left: NormInterpretation, right: NormInterpretation) -> PriorityDecision:
    """Apply explicit legal-priority metadata; never guess missing hierarchy/specificity."""
    if not actions_conflict(left, right):
        return PriorityDecision(None, None, "none", "not_conflicting", "Norms do not express an actionable modal conflict.")

    la, ra = left.authority, right.authority
    if la.authority_rank is not None and ra.authority_rank is not None and la.authority_rank != ra.authority_rank:
        winner, loser = (left, right) if la.authority_rank > ra.authority_rank else (right, left)
        return PriorityDecision(winner.id, loser.id, "lex_superior", "resolved", "Higher authority_rank prevails.")

    left_specific = len(left.conditions) + sum(a == "priority:specific" for a in left.assumptions)
    right_specific = len(right.conditions) + sum(a == "priority:specific" for a in right.assumptions)
    if left_specific != right_specific and max(left_specific, right_specific) > 0:
        winner, loser = (left, right) if left_specific > right_specific else (right, left)
        return PriorityDecision(winner.id, loser.id, "lex_specialis", "resolved", "More specific explicitly represented scope prevails.")

    ld, rd = _date(la.enacted_on), _date(ra.enacted_on)
    if ld is not None and rd is not None and ld != rd and la.authority_rank == ra.authority_rank:
        winner, loser = (left, right) if ld > rd else (right, left)
        return PriorityDecision(winner.id, loser.id, "lex_posterior", "resolved", "Later enactment prevails between equal-ranked sources.")

    return PriorityDecision(None, None, "underdetermined", "unresolved", "Conflict requires legal interpretation: available metadata does not justify lex superior, specialis, or posterior.")


def resolve_conflicts(norms: Sequence[NormInterpretation]) -> List[PriorityDecision]:
    decisions: List[PriorityDecision] = []
    for index, left in enumerate(norms):
        for right in norms[index + 1 :]:
            decision = resolve_pair(left, right)
            if decision.status != "not_conflicting":
                decisions.append(decision)
    return decisions
