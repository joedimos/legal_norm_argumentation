from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

try:
    from .models import load_transformer_classifier, nlp
except ImportError:  # pragma: no cover
    from models import load_transformer_classifier, nlp


_MODAL_RE = re.compile(
    r"\b(shall\s+not|must\s+not|may\s+not|shall|must|may|should|ought\s+to|"
    r"is\s+required\s+to|are\s+required\s+to|is\s+prohibited\s+from|"
    r"are\s+prohibited\s+from|is\s+permitted\s+to|are\s+permitted\s+to|"
    r"is\s+allowed\s+to|are\s+allowed\s+to|"
    r"ne\s+(?:doit|doivent|peut|peuvent)\s+pas|"
    r"ne\s+(?:peut|peuvent)|"
    r"(?:doit|doivent|devra|devront)|(?:peut|peuvent)|"
    r"(?:a|ont)\s+(?:le\s+)?droit\s+de|(?:jouit|jouissent)\s+du\s+droit\s+de|"
    r"(?:est|sont)\s+(?:tenu(?:e|s|es)?\s+de|obligatoire(?:s)?|"
    r"interdit(?:e|s|es)?|prohib(?:é|ée|és|ées)|illicite(?:s)?|autorisé(?:e|s|es)?|"
    r"permis(?:e|es)?)|il\s+est\s+(?:interdit|obligatoire|illicite|permis)\s+de)\b",
    re.IGNORECASE,
)

_CONDITION_RE = re.compile(
    r"^(?P<condition>(?:if|when|whenever|where|provided\s+that|subject\s+to|"
    r"si|lorsque|lorsqu['’]|quand|dès\s+lors\s+que|dans\s+le\s+cas\s+où|"
    r"sous\s+réserve\s+que)\b.+?)(?:,|;)(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)

_EXCEPTION_RE = re.compile(
    r"\b(?P<marker>unless|except(?:\s+where|\s+when|\s+if)?|save\s+where|"
    r"provided\s+however\s+that|sauf(?:\s+si|\s+lorsque|\s+dans\s+le\s+cas\s+où)?|"
    r"à\s+l['’]exception\s+de|hormis|sous\s+réserve\s+de)\b(?P<exception>.+)$",
    re.IGNORECASE | re.DOTALL,
)

_REFERENCE_RE = re.compile(
    r"\b(?:section|article|articles|paragraph|paragraphe|subsection|annex|annexe|schedule|rule|regulation)"
    r"\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9().\-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9().\-]+){0,5}",
    re.IGNORECASE,
)


def _sentences(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield sentence-like spans without losing exact source offsets."""
    for match in re.finditer(r"\S(?:.*?\S)?(?:[.!?](?=\s|$)|$)", text, re.DOTALL):
        raw = match.group(0)
        # Avoid consuming multiple paragraphs as one sentence when punctuation is sparse.
        for part in re.finditer(r"[^\n]+", raw):
            sentence = part.group(0).strip()
            if not sentence:
                continue
            left = len(part.group(0)) - len(part.group(0).lstrip())
            start = match.start() + part.start() + left
            end = start + len(sentence)
            yield sentence, start, end


def classify_norm_type(text: str) -> str:
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "shall not",
            "must not",
            "may not",
            "prohibited",
            "forbidden",
            "ne doit pas",
            "ne doivent pas",
            "ne peut pas",
            "ne peuvent pas",
            "nul ne peut",
            "personne ne peut",
            "personne ne peuvent",
            "interdit",
            "interdite",
            "interdits",
            "interdites",
            "prohibé",
            "prohibée",
            "prohibés",
            "prohibées",
            "illicite",
            "illicites",
        )
    ):
        return "prohibition"
    if any(
        token in lowered
        for token in (
            "shall",
            "must",
            "required",
            "ought to",
            "doit",
            "doivent",
            "devra",
            "devront",
            "est tenu de",
            "est tenue de",
            "sont tenus de",
            "sont tenues de",
            "il est obligatoire de",
        )
    ):
        return "obligation"
    if any(
        token in lowered
        for token in (
            "may",
            "permitted",
            "allowed",
            "peut",
            "peuvent",
            "a le droit de",
            "ont le droit de",
            "a droit de",
            "ont droit de",
            "jouit du droit de",
            "jouissent du droit de",
            "autorisé",
            "autorisée",
            "autorisés",
            "autorisées",
            "est permis",
            "est permise",
            "il est permis de",
        )
    ):
        return "permission"
    if any(token in lowered for token in ("should", "devrait", "devraient")):
        return "recommendation"
    return "unknown"


def _split_scope(sentence: str, modal_match: re.Match[str]) -> tuple[str | None, str]:
    actor = sentence[: modal_match.start()].strip(" ,;:") or None
    action = sentence[modal_match.end() :].strip(" ,;:.")
    return actor, action


def _stable_id(source_id: str, start: int, end: int, sentence: str) -> str:
    digest = hashlib.sha256(f"{source_id}:{start}:{end}:{sentence}".encode("utf-8")).hexdigest()[:16]
    return f"norm-{digest}"


def extract_norms(
    text: str,
    use_transformer: bool = False,
    source_id: str = "inline",
) -> list[dict]:
    """Extract reviewable English/French norm candidates with exact provenance.

    The function does not claim that a candidate is a faithful formalisation.
    It preserves conditions/exceptions and surfaces ambiguity for later review.
    """
    classifier = load_transformer_classifier() if use_transformer else None
    results: list[dict] = []

    for sentence, start, end in _sentences(text):
        modal = _MODAL_RE.search(sentence)
        if not modal:
            continue

        condition = None
        scoped_sentence = sentence
        condition_match = _CONDITION_RE.match(sentence)
        if condition_match:
            condition = condition_match.group("condition").strip()
            scoped_sentence = condition_match.group("body").strip()
            modal = _MODAL_RE.search(scoped_sentence) or modal

        exception = None
        exception_match = _EXCEPTION_RE.search(scoped_sentence)
        if exception_match:
            exception = (exception_match.group("marker") + exception_match.group("exception")).strip(" .")

        actor, action = _split_scope(scoped_sentence, modal)
        if exception_match:
            after_modal = scoped_sentence[modal.end() :]
            exc_in_after = _EXCEPTION_RE.search(after_modal)
            if exc_in_after:
                action = after_modal[: exc_in_after.start()].strip(" ,;:.")

        modality = classify_norm_type(scoped_sentence)
        # French codes often express an unlawful act as "X est illicite/interdit".
        # Preserve X as the action while leaving the actor unresolved for human review.
        if modality == "prohibition" and not action and actor and re.match(r"^(?:est|sont)\s+", modal.group(0), re.I):
            action = actor
            actor = None

        ambiguity_flags: list[str] = []
        if actor is None:
            ambiguity_flags.append("missing_actor")
        if not action:
            ambiguity_flags.append("missing_action")
        if re.search(
            r"\b(it|they|this|that|such|thereof|therein|celui-ci|celle-ci|ceux-ci|celles-ci|"
            r"ledit|ladite|lesdits|lesdites)\b",
            sentence,
            re.IGNORECASE,
        ):
            ambiguity_flags.append("possible_anaphora")
        if modality == "unknown":
            ambiguity_flags.append("unknown_modality")

        confidence = 0.72
        classifier_label = None
        if classifier is not None:
            try:
                prediction = classifier(sentence)
                if prediction:
                    confidence = float(prediction[0].get("score", confidence))
                    classifier_label = prediction[0].get("label")
            except Exception:
                ambiguity_flags.append("classifier_failed")

        results.append(
            {
                "id": _stable_id(source_id, start, end, sentence),
                "source_id": source_id,
                "source_span": [start, end],
                "sentence": sentence,
                "text": sentence,
                "modal": modal.group(0),
                "modality": modality,
                "type": modality,
                "actor": actor,
                "action": action,
                "condition": condition,
                "exception": exception,
                "references": _REFERENCE_RE.findall(sentence),
                "ambiguity_flags": sorted(set(ambiguity_flags)),
                "confidence": confidence,
                "classifier_label": classifier_label,
                "review_status": "unreviewed",
            }
        )

    return results


def extract_deontic_operators(text: str) -> list[dict]:
    return [
        {
            "type": norm["modality"],
            "operator": {
                "obligation": "O",
                "permission": "P",
                "prohibition": "F",
                "recommendation": "R",
            }.get(norm["modality"], "?"),
            "pattern": norm["modal"],
            "text": norm["sentence"],
        }
        for norm in extract_norms(text)
    ]


def extract_conditional_norms(text: str) -> list[dict]:
    return [
        {
            "condition": norm["condition"],
            "consequence": norm["action"],
            "full_text": norm["sentence"],
            "type": "conditional",
        }
        for norm in extract_norms(text)
        if norm["condition"]
    ]


def extract_legal_entities(text: str) -> list[dict]:
    if nlp is None:
        return []
    doc = nlp(text)
    return [
        {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "LAW"}
    ]


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9à-öø-ÿ]+", value.lower()) if len(token) > 2}


def has_similar_content(text1: str, text2: str, threshold: float = 0.5) -> bool:
    left, right = _token_set(text1), _token_set(text2)
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= threshold


def analyze_norm_conflicts(norms: list[dict]) -> list[dict]:
    conflicts: list[dict] = []
    for i, left in enumerate(norms):
        for right in norms[i + 1 :]:
            modalities = {left.get("modality") or left.get("type"), right.get("modality") or right.get("type")}
            if modalities != {"obligation", "prohibition"}:
                continue
            same_actor = (left.get("actor") or "").lower() == (right.get("actor") or "").lower()
            similar_action = has_similar_content(left.get("action", ""), right.get("action", ""), threshold=0.35)
            if same_actor and similar_action:
                conflicts.append(
                    {
                        "norm1_id": left.get("id"),
                        "norm2_id": right.get("id"),
                        "type": "obligation-prohibition",
                        "severity": "high",
                        "requires_legal_priority_analysis": True,
                    }
                )
    return conflicts


def extract_all_norms(text: str, use_transformer: bool = False, source_id: str = "inline") -> dict:
    norms = extract_norms(text, use_transformer=use_transformer, source_id=source_id)
    return {
        "norms": norms,
        "deontic_operators": extract_deontic_operators(text),
        "conditionals": extract_conditional_norms(text),
        "entities": extract_legal_entities(text),
        "conflicts": analyze_norm_conflicts(norms),
    }
