"""Source snapshots and content-bound, self-reported research review receipts.

A matching quote/hash proves integrity, not the truth of a factual assertion or
faithfulness of a legal interpretation. This module does not authenticate people.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime

from .audit_bundle import canonical_json, digest
from .temporal import Formula, validate_formula, validate_trace

FIELDS = {"schema", "context", "documents", "interpretations", "horizon", "observations", "grounding",
          "reviews", "queries", "max_completions", "max_worlds", "max_states", "timeout_ms"}


def _text(value, label, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} requires 1-{limit} characters")


def _date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def family_fingerprint(request):
    return digest(canonical_json({"context": request["context"], "interpretations": request["interpretations"],
                                 "horizon": request["horizon"],
                                 "documents": [d for d in request["documents"] if d["kind"] == "normative"]}))


def grounding_fingerprint(request):
    return digest(canonical_json({"family": family_fingerprint(request),
                                 "observations": request.get("observations"), "grounding": request.get("grounding", []),
                                 "documents": [d for d in request["documents"] if d["kind"] != "normative"]}))


def parse_family(request):
    if (not isinstance(request, dict) or request.get("schema") != "vle-family/v1" or set(request) - FIELDS
            or not {"context", "documents", "interpretations", "horizon"} <= set(request)):
        raise ValueError("invalid vle-family/v1 request")
    context = request["context"]
    if not isinstance(context, dict) or set(context) != {"jurisdiction", "as_of", "actor", "capacity"}:
        raise ValueError("context requires jurisdiction, as_of, actor, and capacity")
    for key in ("jurisdiction", "actor", "capacity"):
        _text(context[key], key)
    current = _date(context["as_of"])
    horizon = request["horizon"]
    if type(horizon) is not int or not 1 <= horizon <= 128:
        raise ValueError("horizon must be 1-128")
    findings = []
    documents = request["documents"]
    if not isinstance(documents, list) or not 1 <= len(documents) <= 32:
        raise ValueError("documents requires 1-32 snapshots")
    registry = {}
    for document in documents:
        required = {"id", "version", "kind", "language", "text", "sha256"}
        if (not isinstance(document, dict) or not required <= set(document)
                or set(document) - (required | {"jurisdiction", "effective_from", "effective_to", "references"})):
            raise ValueError("invalid document snapshot fields")
        for key in ("id", "version", "language"):
            _text(document[key], key)
        if document["id"] in registry:
            raise ValueError("snapshot ids must be unique; select a version explicitly")
        if not isinstance(document["kind"], str) or document["kind"] not in {"normative", "observation", "draft"}:
            raise ValueError("unsupported document kind")
        _text(document["text"], "document text", 500000)
        if document["sha256"] != digest(document["text"].encode("utf-8")):
            raise ValueError("snapshot hash does not match its text")
        start, end = document.get("effective_from"), document.get("effective_to")
        start = _date(start) if start is not None else None
        end = _date(end) if end is not None else None
        if start and end and start > end:
            raise ValueError("effective_from must not follow effective_to")
        if document["kind"] == "normative":
            if not {"effective_from", "effective_to"} <= set(document):
                findings.append({"code": "unknown_effective_period", "document_id": document["id"]})
            if document.get("jurisdiction") != context["jurisdiction"]:
                findings.append({"code": "jurisdiction_review_required", "document_id": document["id"]})
            if (start and current < start) or (end and current > end):
                findings.append({"code": "ineffective_source", "document_id": document["id"]})
        references = document.get("references", [])
        if not isinstance(references, list) or len(references) > 32 or any(not isinstance(r, str) for r in references):
            raise ValueError("document references must be a bounded string array")
        registry[document["id"]] = document
    for document in documents:
        for target in document.get("references", []):
            if target not in registry:
                findings.append({"code": "unresolved_reference", "document_id": document["id"], "target": target})
            elif document["kind"] == "normative" and registry[target]["kind"] != "normative":
                findings.append({"code": "non_normative_reference", "document_id": document["id"], "target": target})

    def references(values, kind):
        if not isinstance(values, list) or not 1 <= len(values) <= 32:
            raise ValueError("evidence references require 1-32 quote spans")
        for ref in values:
            if not isinstance(ref, dict) or set(ref) != {"document_id", "start", "end", "quote"}:
                raise ValueError("invalid evidence reference")
            _text(ref["document_id"], "reference document id")
            document = registry.get(ref["document_id"])
            if document is None:
                raise ValueError("evidence reference has no snapshot")
            start, end = ref["start"], ref["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document["text"]):
                raise ValueError("invalid quote offsets")
            if document["text"][start:end] != ref["quote"]:
                raise ValueError("quote does not match its snapshot")
            if document["kind"] != kind:
                findings.append({"code": "untrusted_grounding" if kind == "observation" else "non_normative_source",
                                 "document_id": document["id"]})

    interpretations = request["interpretations"]
    if not isinstance(interpretations, list) or not 1 <= len(interpretations) <= 16:
        raise ValueError("interpretations requires 1-16 members")
    parsed, seen, names = [], set(), set()
    for item in interpretations:
        if (not isinstance(item, dict) or not {"id", "property", "source_refs", "rationale"} <= set(item)
                or set(item) - {"id", "property", "source_refs", "rationale", "assumptions"}):
            raise ValueError("invalid interpretation fields")
        _text(item["id"], "interpretation id")
        _text(item["rationale"], "interpretation rationale", 4000)
        if item["id"] in seen:
            raise ValueError("duplicate interpretation id")
        seen.add(item["id"])
        references(item["source_refs"], "normative")
        property = Formula.from_dict(item["property"])
        names |= validate_formula(property)
        assumptions = item.get("assumptions", [])
        if not isinstance(assumptions, list) or len(assumptions) > 64:
            raise ValueError("at most 64 assumptions per interpretation")
        assumptions = [Formula.from_dict(a) for a in assumptions]
        for formula in assumptions:
            names |= validate_formula(formula)
        parsed.append({"id": item["id"], "property": property, "assumptions": assumptions})
    observed = request.get("observations")
    if observed is not None:
        if not isinstance(observed, list):
            raise ValueError("observations must be an array")
        validate_trace(observed)
        if len(observed) > horizon:
            raise ValueError("observations extend beyond horizon")
        for state in observed:
            for name in state:
                _text(name, "observation atom")
                names.add(name)
    fixed = {(t, name): value for t, state in enumerate(observed or ()) for name, value in state.items()
             if value is not None}
    grounding = request.get("grounding", [])
    if not isinstance(grounding, list) or len(grounding) > 8192:
        raise ValueError("grounding must be an array of at most 8192 records")
    grounded = set()
    for item in grounding:
        if not isinstance(item, dict) or set(item) != {"step", "atom", "value", "evidence_refs"}:
            raise ValueError("invalid grounding record")
        if type(item["step"]) is not int or not isinstance(item["atom"], str) or type(item["value"]) is not bool:
            raise ValueError("invalid grounding cell/value")
        cell = item["step"], item["atom"]
        if cell in grounded or cell not in fixed or fixed[cell] != item["value"]:
            raise ValueError("grounding must uniquely match an observed Boolean cell")
        grounded.add(cell)
        references(item["evidence_refs"], "observation")
    for t, name in sorted(fixed.keys() - grounded):
        findings.append({"code": "missing_grounding", "step": t, "atom": name})
    reviews = request.get("reviews", {})
    if not isinstance(reviews, dict) or set(reviews) - {"family", "grounding"}:
        raise ValueError("invalid review scopes")
    for scope, expected in (("family", family_fingerprint(request)), ("grounding", grounding_fingerprint(request))):
        if scope == "grounding" and not fixed and scope not in reviews:
            continue
        record = reviews.get(scope)
        if record is None:
            findings.append({"code": "unreviewed", "scope": scope})
            continue
        if not isinstance(record, dict) or set(record) != {"decision", "reviewer", "reviewed_at", "notes", "fingerprint"}:
            raise ValueError("invalid review receipt")
        for key in ("reviewer", "notes", "reviewed_at", "fingerprint", "decision"):
            _text(record[key], key, 4000)
        if datetime.fromisoformat(record["reviewed_at"]).tzinfo is None:
            raise ValueError("review timestamps require timezone")
        if record["decision"] not in {"approved", "rejected", "needs_revision"}:
            raise ValueError("invalid review decision")
        if record["fingerprint"] != expected or record["decision"] != "approved":
            findings.append({"code": "stale_or_unapproved_review", "scope": scope})
    return {"interpretations": parsed, "names": names, "fixed": fixed, "findings": findings}


def record_family_review(request, *, scope, reviewer, notes, decision="approved"):
    """Explicit lab receipt; caller identity is self-reported, not authenticated."""
    parse_family(request)
    if scope not in {"family", "grounding"} or decision not in {"approved", "rejected", "needs_revision"}:
        raise ValueError("invalid review scope/decision")
    _text(reviewer, "reviewer")
    _text(notes, "review notes", 4000)
    result = deepcopy(request)
    fingerprint = family_fingerprint(result) if scope == "family" else grounding_fingerprint(result)
    result.setdefault("reviews", {})[scope] = {"decision": decision, "reviewer": reviewer, "notes": notes,
                                              "reviewed_at": datetime.now(UTC).isoformat(),
                                              "fingerprint": fingerprint}
    return result
