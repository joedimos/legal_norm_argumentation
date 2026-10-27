"""Blinded independent annotation packets and disagreement-preserving assessment.

Identity, qualifications and independence are attestations, not authenticated
credentials. Assessment never creates family/grounding/behavior approval receipts.
"""
from __future__ import annotations

from copy import deepcopy

from .audit_bundle import canonical_json, digest
from .family_contract import parse_family

DIMENSIONS = ("applicability", "faithfulness", "conditions_exceptions", "grounding", "response_behavior",
              "harmful_assistance", "usefulness", "refusal_appropriateness")
LABELS = {"supported", "unsupported", "uncertain", "not_applicable"}


def _text(value, name, limit=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid {name}")


def _hash(value):
    return digest(canonical_json(value))


def prepare_annotation(request):
    if (not isinstance(request, dict) or request.get("schema") != "vle-annotation-study/v1"
            or set(request) != {"schema", "study_id", "units"}):
        raise ValueError("invalid annotation study")
    _text(request["study_id"], "study_id", 128)
    units = request["units"]
    if not isinstance(units, list) or not 1 <= len(units) <= 200:
        raise ValueError("study requires 1-200 units")
    registry, groups, snapshots, output = set(), {}, {}, []
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {"id", "source_group", "split", "prompt", "family"}:
            raise ValueError("unit requires id, source_group, split, prompt, family")
        for key in ("id", "source_group", "prompt"):
            _text(unit[key], key, 32000 if key == "prompt" else 128)
        if unit["id"] in registry:
            raise ValueError("duplicate annotation unit")
        registry.add(unit["id"])
        if unit["split"] not in {"train", "development", "test"}:
            raise ValueError("invalid study split")
        group = unit["source_group"]
        if group in groups and groups[group] != unit["split"]:
            raise ValueError("source/amendment group crosses study splits")
        groups[group] = unit["split"]
        parse_family(unit["family"])
        for document in unit["family"]["documents"]:
            if document["kind"] == "normative":
                sha = document["sha256"]
                if sha in snapshots and snapshots[sha] != unit["split"]:
                    raise ValueError("normative snapshot crosses study splits despite source_group labels")
                snapshots[sha] = unit["split"]
        # Strict unit schema excludes predictions, model identity and solver output.
        # Human study designers must additionally scrub clues from source text.
        blinded = deepcopy(unit)
        blinded["family"].pop("reviews", None)
        output.append({**blinded, "unit_sha256": _hash(blinded)})
    body = {"schema": "vle-annotation-packet/v1", "study_id": request["study_id"],
            "dimensions": list(DIMENSIONS), "labels": sorted(LABELS), "units": output}
    return {**body, "packet_sha256": _hash(body), "annotation_status": "awaiting_independent_annotations"}


def _validate_packet(packet):
    if not isinstance(packet, dict) or set(packet) != {"schema", "study_id", "dimensions", "labels", "units",
                                                     "packet_sha256", "annotation_status"}:
        raise ValueError("invalid annotation packet")
    if packet["schema"] != "vle-annotation-packet/v1":
        raise ValueError("unsupported annotation packet")
    units = packet["units"]
    if not isinstance(units, list) or not units:
        raise ValueError("empty annotation packet")
    original = {"schema": "vle-annotation-study/v1", "study_id": packet["study_id"],
                "units": [{k: v for k, v in unit.items() if k != "unit_sha256"} for unit in units]}
    if prepare_annotation(original) != packet:
        raise ValueError("annotation packet content/hash mismatch")


def _reviewer(value):
    if (not isinstance(value, dict) or set(value) != {"id", "qualification", "independent", "conflicts_disclosed"}
            or value["independent"] is not True or value["conflicts_disclosed"] is not False):
        raise ValueError("reviewer must attest independence and no conflicts for this study")
    _text(value["id"], "reviewer id", 128)
    _text(value["qualification"], "reviewer qualification")


def _judgment(value):
    if (not isinstance(value, dict) or set(value) != {"labels", "rationale", "alternative_interpretations"}
            or not isinstance(value["labels"], dict) or set(value["labels"]) != set(DIMENSIONS)):
        raise ValueError("judgment requires all separate dimensions, rationale, alternative_interpretations")
    if any(not isinstance(v, str) or v not in LABELS for v in value["labels"].values()):
        raise ValueError("invalid annotation label")
    _text(value["rationale"], "judgment rationale", 16000)
    alternatives = value["alternative_interpretations"]
    if not isinstance(alternatives, list) or len(alternatives) > 16:
        raise ValueError("alternative interpretations must be a bounded array")
    for alternative in alternatives:
        _text(alternative, "alternative interpretation", 8000)


def assess_annotations(request):
    if (not isinstance(request, dict) or request.get("schema") != "vle-annotation-assessment/v1"
            or not {"schema", "packet", "submissions"} <= set(request)
            or set(request) - {"schema", "packet", "submissions", "adjudication"}):
        raise ValueError("invalid annotation assessment")
    packet = request["packet"]
    _validate_packet(packet)
    submissions = request["submissions"]
    if not isinstance(submissions, list) or len(submissions) != 2:
        raise ValueError("exactly two independent submissions required")
    units = {unit["id"]: unit for unit in packet["units"]}
    reviewer_ids, indexed = set(), []
    for submission in submissions:
        if (not isinstance(submission, dict) or set(submission) != {"reviewer", "packet_sha256", "annotations"}
                or submission["packet_sha256"] != packet["packet_sha256"]):
            raise ValueError("submission must bind the frozen packet")
        _reviewer(submission["reviewer"])
        id = submission["reviewer"]["id"]
        if id in reviewer_ids:
            raise ValueError("reviewers must have distinct identities")
        reviewer_ids.add(id)
        annotations = submission["annotations"]
        if not isinstance(annotations, list) or len(annotations) != len(units):
            raise ValueError("each reviewer must annotate every unit")
        by_id = {}
        for annotation in annotations:
            if not isinstance(annotation, dict) or set(annotation) != {"unit_id", "unit_sha256", "judgment"}:
                raise ValueError("invalid annotation")
            id = annotation["unit_id"]
            if id not in units or id in by_id or annotation["unit_sha256"] != units[id]["unit_sha256"]:
                raise ValueError("unknown, duplicate or stale annotation unit")
            _judgment(annotation["judgment"])
            by_id[id] = annotation["judgment"]
        indexed.append(by_id)
    differences = {}
    metrics = {}
    n = len(units)
    for dimension in DIMENSIONS:
        a = [indexed[0][id]["labels"][dimension] for id in units]
        b = [indexed[1][id]["labels"][dimension] for id in units]
        agreement = sum(x == y for x, y in zip(a, b, strict=True)) / n
        expected = sum(a.count(label) * b.count(label) for label in LABELS) / (n * n)
        metrics[dimension] = {"units": n, "agreement": agreement,
                              "cohen_kappa": (agreement - expected) / (1 - expected) if expected < 1 else None,
                              "kappa_undefined_reason": "constant_identical_labels" if expected == 1 else None}
    for id in units:
        dimensions = [d for d in DIMENSIONS if indexed[0][id]["labels"][d] != indexed[1][id]["labels"][d]]
        # Same labels do not establish agreement about interpretation content.
        if indexed[0][id]["alternative_interpretations"] != indexed[1][id]["alternative_interpretations"]:
            dimensions.append("alternative_interpretations")
        if dimensions:
            differences[id] = dimensions
    adjudication = request.get("adjudication")
    resolved = set()
    if adjudication is not None:
        if (not isinstance(adjudication, dict) or set(adjudication) != {"reviewer", "submission_sha256s", "decisions"}
                or adjudication["submission_sha256s"] != [_hash(s) for s in submissions]):
            raise ValueError("adjudication must bind ordered original submissions")
        _reviewer(adjudication["reviewer"])
        if adjudication["reviewer"]["id"] in reviewer_ids:
            raise ValueError("adjudicator must be a third reviewer")
        decisions = adjudication["decisions"]
        if not isinstance(decisions, list) or len(decisions) > len(differences):
            raise ValueError("invalid adjudication decisions")
        seen = set()
        for decision in decisions:
            if not isinstance(decision, dict) or set(decision) != {"unit_id", "resolution", "judgment"}:
                raise ValueError("invalid adjudication decision")
            id = decision["unit_id"]
            if id not in differences or id in seen:
                raise ValueError("duplicate or undisputed adjudication unit")
            seen.add(id)
            if decision["resolution"] not in {"resolved", "unresolved"}:
                raise ValueError("adjudication resolution must be resolved or unresolved")
            _judgment(decision["judgment"])
            if decision["resolution"] == "resolved":
                resolved.add(id)
    return {"schema": "vle-annotation-result/v1", "request_sha256": _hash(request),
            "status": "adjudication_required" if set(differences) - resolved else "annotations_collected",
            "packet_sha256": packet["packet_sha256"], "submissions": deepcopy(submissions),
            "adjudication": deepcopy(adjudication), "disagreements": differences,
            "unresolved_units": sorted(set(differences) - resolved), "metrics": metrics,
            "reviewers_authenticated": False, "qualifications_verified": False, "independence_verified": False,
            "deployment_authorized": False, "legal_certification": False}
