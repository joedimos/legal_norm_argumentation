"""Versioned JSON entry points for reviewed research inputs and experiments."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

from .conflicts import detect_temporal_conflicts
from .normative_model import Modality, NormInterpretation, Predicate, SourceAuthority, SourceKind
from .preferences import Preference, analyze_preferences
from .temporal_deontic import TemporalNorm


def norm_from_dict(data: Mapping) -> NormInterpretation:
    if not isinstance(data, Mapping) or set(data) - {
        "id", "modality", "subject", "action", "object", "conditions", "exceptions", "authority"
    }:
        raise ValueError("unsupported norm fields")
    if not {"id", "modality", "subject", "action"} <= set(data):
        raise ValueError("norm requires id, modality, subject and action")
    if any(not isinstance(data[k], str) or not data[k].strip() for k in ("id", "subject", "action")):
        raise ValueError("norm identifiers and target fields must be non-empty strings")
    if data.get("object") is not None and not isinstance(data["object"], str):
        raise ValueError("object must be a string or null")

    def predicates(key):
        values = data.get(key, [])
        if not isinstance(values, list) or len(values) > 32:
            raise ValueError("predicate lists are limited to 32 entries")
        result = []
        for value in values:
            if (not isinstance(value, Mapping) or set(value) - {"name", "negated"}
                    or not isinstance(value.get("name"), str) or type(value.get("negated", False)) is not bool):
                raise ValueError("invalid predicate")
            result.append(Predicate(value["name"], value.get("negated", False)))
        return tuple(result)

    authority = data.get("authority", {})
    if not isinstance(authority, Mapping) or set(authority) - {f.name for f in fields(SourceAuthority)}:
        raise ValueError("invalid authority fields")
    authority = dict(authority)
    if "kind" in authority:
        authority["kind"] = SourceKind(authority["kind"])
    return NormInterpretation(id=data["id"], source_norm_id=data["id"], modality=Modality(data["modality"]),
                              subject=data["subject"], action=data["action"], object=data.get("object"),
                              conditions=predicates("conditions"), exceptions=predicates("exceptions"),
                              authority=SourceAuthority(**authority))


def temporal_norm_from_dict(data: Mapping) -> TemporalNorm:
    if not isinstance(data, Mapping) or set(data) - {"norm", "start", "end", "mode", "activation_date"}:
        raise ValueError("invalid temporal norm fields")
    if not {"norm", "start", "end"} <= set(data):
        raise ValueError("temporal norm requires norm, start and end")
    return TemporalNorm(norm_from_dict(data["norm"]), data["start"], data["end"],
                        data.get("mode", "achievement"), data.get("activation_date"))


def run_research(payload: Mapping) -> dict:
    """Temporal findings + normative conflicts + explicit preference reasoning.

    Unknown applicability remains unknown. Potential partial-window conflicts do
    not create defeat edges; jointly satisfiable duties are retained.
    """
    if not isinstance(payload, Mapping) or set(payload) - {"schema", "norms", "trace", "closed", "preferences"}:
        raise ValueError("invalid research request fields")
    if payload.get("schema") != "vle-research/v1":
        raise ValueError("unsupported research schema")
    if not isinstance(payload.get("norms"), list) or not 1 <= len(payload["norms"]) <= 64:
        raise ValueError("research request requires 1-64 norms")
    if not isinstance(payload.get("trace"), list):
        raise ValueError("research request requires an observed trace")
    norms = [temporal_norm_from_dict(item) for item in payload["norms"]]
    if len({n.norm.id for n in norms}) != len(norms):
        raise ValueError("norm ids must be unique")
    findings = [n.evaluate(payload["trace"], closed=payload.get("closed", True)) for n in norms]
    by_id = {n.norm.id: n for n in norms}
    # Evaluate applicability independently of fulfilment and window completion.
    from .temporal import evaluate_formula

    active_ids = {n.norm.id for n in norms if len(payload["trace"]) > n.start
                  and evaluate_formula(n.guard(), payload["trace"], n.start) is True}
    conflicts = detect_temporal_conflicts([by_id[i] for i in sorted(active_ids)])
    attacks = set()
    for conflict in conflicts:
        if conflict.severity == "incompatible":
            attacks.update({(conflict.left_id, conflict.right_id), (conflict.right_id, conflict.left_id)})
    raw_preferences = payload.get("preferences", [])
    if not isinstance(raw_preferences, list) or len(raw_preferences) > 256:
        raise ValueError("preferences must be a list of at most 256 entries")
    preferences = []
    for entry in raw_preferences:
        if not isinstance(entry, dict) or set(entry) != {"preferred", "disfavored", "rationale"}:
            raise ValueError("invalid preference fields")
        if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
            raise ValueError("preference fields must be non-empty strings")
        preferences.append(Preference(**entry))
    # Validate the entire order, including inactive norms, before restricting it.
    full = analyze_preferences(by_id, attacks, preferences)
    # Transitive preferences may legitimately be mediated by inactive norms;
    # derive the active defeat graph from the validated full order.
    argumentation = analyze_preferences(sorted(active_ids), {tuple(e) for e in full["defeats"]})
    argumentation["suppressed_attacks"] = full["suppressed_attacks"]
    labels = argumentation["grounded_labels"]
    effective = []
    for item in findings:
        label = labels.get(item["norm_id"])
        status = "defeated" if label == "out" else "conflicted" if label == "undec" else item["status"]
        effective.append({**item, "status": status, "raw_status": item["status"]})
    states = {item["status"] for item in effective}
    status = ("indeterminate" if states & {"unknown", "pending", "conflicted"}
              else "non_compliant" if "violated" in states else "compliant")
    return {"schema": "vle-research-result/v1", "status": status, "findings": effective,
            "conflicts": [c.to_dict() for c in conflicts], "argumentation": argumentation,
            "legal_certification": False, "preferences": raw_preferences,
            "semantics": "single-activation-inclusive-discrete-windows-v1"}
