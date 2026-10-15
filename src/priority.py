from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .conflicts import scope_literals, scopes_overlap
from .normative_model import NormInterpretation


@dataclass(frozen=True)
class PriorityDecision:
    winner_id: str | None
    loser_id: str | None
    principle: str
    status: str
    explanation: str


def actions_conflict(left: NormInterpretation, right: NormInterpretation) -> bool:
    if not scopes_overlap(left, right):
        return False
    pair = {left.modality.value, right.modality.value}
    return pair in ({"obligation", "prohibition"}, {"permission", "prohibition"})


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def resolve_pair(left: NormInterpretation, right: NormInterpretation) -> PriorityDecision:
    """Apply explicit legal-priority metadata; never guess missing hierarchy/specificity."""
    if not actions_conflict(left, right):
        return PriorityDecision(None, None, "none", "not_conflicting", "Norms do not express an actionable modal conflict.")

    la, ra = left.authority, right.authority
    if la.authority_rank is not None and ra.authority_rank is not None and la.authority_rank != ra.authority_rank:
        winner, loser = (left, right) if la.authority_rank > ra.authority_rank else (right, left)
        return PriorityDecision(winner.id, loser.id, "lex_superior", "resolved", "Higher authority_rank prevails.")

    left_scope, right_scope = scope_literals(left), scope_literals(right)
    if left_scope > right_scope or right_scope > left_scope:
        winner, loser = (left, right) if left_scope > right_scope else (right, left)
        return PriorityDecision(winner.id, loser.id, "lex_specialis", "resolved", "Strict conjunctive scope inclusion establishes specificity.")

    ld, rd = _date(la.enacted_on), _date(ra.enacted_on)
    if ld is not None and rd is not None and ld != rd and la.authority_rank is not None and la.authority_rank == ra.authority_rank:
        winner, loser = (left, right) if ld > rd else (right, left)
        return PriorityDecision(winner.id, loser.id, "lex_posterior", "resolved", "Later enactment prevails between equal-ranked sources.")

    return PriorityDecision(None, None, "underdetermined", "unresolved", "Conflict requires legal interpretation: available metadata does not justify lex superior, specialis, or posterior.")


def resolve_conflicts(norms: Sequence[NormInterpretation]) -> list[PriorityDecision]:
    decisions: list[PriorityDecision] = []
    for index, left in enumerate(norms):
        for right in norms[index + 1 :]:
            decision = resolve_pair(left, right)
            if decision.status != "not_conflicting":
                decisions.append(decision)
    return decisions
