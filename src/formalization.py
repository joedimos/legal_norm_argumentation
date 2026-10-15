from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from re import I, compile

from .normative_model import (
    InterpretationSet,
    Modality,
    NormInterpretation,
    Predicate,
    SourceAuthority,
    SourceProvenance,
    stable_interpretation_id,
)


_CONDITION_PREFIX_RE = compile(
    r"^(?:(?:if|when|whenever|where|provided\s+that|subject\s+to|"
    r"si|lorsque|quand|dès\s+lors\s+que|dans\s+le\s+cas\s+où|sous\s+réserve\s+que)\s+|"
    r"lorsqu['’])",
    flags=I,
)

_EXCEPTION_PREFIX_RE = compile(
    r"^(?:unless|except(?:\s+where|\s+when|\s+if)?|save\s+where|provided\s+however\s+that|"
    r"sauf(?:\s+si|\s+lorsque|\s+dans\s+le\s+cas\s+où)?|à\s+l['’]exception\s+de|hormis|"
    r"sous\s+réserve\s+de)\s+",
    flags=I,
)


def _predicate_from_clause(clause: str | None) -> tuple[Predicate, ...]:
    if not clause:
        return ()
    cleaned = _CONDITION_PREFIX_RE.sub("", clause).strip(" ,.;")
    return (Predicate(cleaned),) if cleaned else ()


def _exception_predicate(clause: str | None) -> tuple[Predicate, ...]:
    if not clause:
        return ()
    cleaned = _EXCEPTION_PREFIX_RE.sub("", clause).strip(" ,.;")
    return (Predicate(cleaned),) if cleaned else ()


def candidate_to_interpretations(
    candidate: Mapping,
    authority: SourceAuthority | None = None,
) -> InterpretationSet:
    source_norm_id = str(candidate.get("id") or "")
    if not source_norm_id:
        raise ValueError("candidate must have an id")
    modality_text = str(candidate.get("modality") or "unknown")
    try:
        modality = Modality(modality_text)
    except ValueError as exc:
        raise ValueError(f"cannot formalize unknown modality: {modality_text}") from exc
    subject = (candidate.get("actor") or "").strip()
    action = (candidate.get("action") or "").strip()
    if not subject or not action:
        raise ValueError("candidate requires actor and action before formalisation")
    span = candidate.get("source_span") or [0, 0]
    provenance = SourceProvenance(
        source_id=str(candidate.get("source_id") or "inline"),
        span=(int(span[0]), int(span[1])),
        quote=str(candidate.get("sentence") or candidate.get("text") or ""),
        document_hash=candidate.get("document_hash"),
        locator=candidate.get("locator"),
    )
    assumptions = tuple(f"review:{flag}" for flag in candidate.get("ambiguity_flags", []) if flag)
    base = NormInterpretation(
        id="pending",
        source_norm_id=source_norm_id,
        modality=modality,
        subject=subject,
        action=action,
        conditions=_predicate_from_clause(candidate.get("condition")),
        exceptions=_exception_predicate(candidate.get("exception")),
        references=tuple(candidate.get("references") or ()),
        authority=authority or SourceAuthority(),
        confidence=float(candidate.get("confidence", 1.0)),
        assumptions=assumptions,
        provenance=provenance,
    )
    primary = replace(base, id=stable_interpretation_id(source_norm_id, "primary", base.canonical()))
    return InterpretationSet(
        source_norm_id=source_norm_id,
        interpretations=(primary,),
        selected_id=primary.id if not assumptions else None,
        selection_rationale="deterministic conservative compilation" if not assumptions else None,
    )


def formalize_candidates(
    candidates: Sequence[Mapping],
    authorities: Mapping[str, SourceAuthority] | None = None,
) -> list[InterpretationSet]:
    authorities = authorities or {}
    return [candidate_to_interpretations(c, authorities.get(str(c.get("source_id")))) for c in candidates]


def add_alternative(
    interpretation_set: InterpretationSet,
    *,
    label: str,
    subject: str | None = None,
    action: str | None = None,
    conditions: Sequence[Predicate] | None = None,
    exceptions: Sequence[Predicate] | None = None,
    assumptions: Sequence[str] = (),
    confidence: float | None = None,
) -> InterpretationSet:
    if not interpretation_set.interpretations:
        raise ValueError("cannot extend an empty interpretation set")
    base = interpretation_set.interpretations[0]
    draft = replace(
        base,
        id="pending",
        subject=subject or base.subject,
        action=action or base.action,
        conditions=tuple(conditions) if conditions is not None else base.conditions,
        exceptions=tuple(exceptions) if exceptions is not None else base.exceptions,
        assumptions=tuple(assumptions),
        confidence=base.confidence if confidence is None else confidence,
        interpretation_label=label,
    )
    draft = replace(draft, id=stable_interpretation_id(base.source_norm_id, label, draft.canonical()))
    return InterpretationSet(interpretation_set.source_norm_id, interpretation_set.interpretations + (draft,), None, None)


def select_interpretation(
    interpretation_set: InterpretationSet,
    interpretation_id: str,
    rationale: str,
) -> InterpretationSet:
    if not any(item.id == interpretation_id for item in interpretation_set.interpretations):
        raise ValueError("selected interpretation does not exist")
    if not rationale.strip():
        raise ValueError("selection rationale is required")
    return InterpretationSet(
        interpretation_set.source_norm_id,
        interpretation_set.interpretations,
        interpretation_id,
        rationale.strip(),
    )


def behavioral_signature(interpretation: NormInterpretation, contexts: Iterable[Mapping[str, bool]]) -> tuple:
    signature = []
    for facts in contexts:
        cond = tuple(predicate.evaluate(facts) for predicate in interpretation.conditions)
        exc = tuple(predicate.evaluate(facts) for predicate in interpretation.exceptions)
        signature.append((cond, exc, interpretation.modality.value, interpretation.subject, interpretation.action))
    return tuple(signature)


def behaviorally_equivalent(
    left: NormInterpretation,
    right: NormInterpretation,
    contexts: Iterable[Mapping[str, bool]],
) -> bool:
    contexts = list(contexts)
    return behavioral_signature(left, contexts) == behavioral_signature(right, contexts)
