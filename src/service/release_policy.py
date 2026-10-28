"""Fail-closed organizational release gates, separate from research readiness.

Externally pinned evidence is a custodian's release attestation. Hashes establish
integrity, not the truth of an experiment or the legal authority of an expert.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from src.audit_bundle import MAX_BYTES, canonical_json, digest
from src.family_contract import parse_family
from src.model_runtime import strict_json
from src.signed_annotation import assess_signed_annotations

HASH = re.compile(r"[0-9a-f]{64}\Z")


def load_json_file(path):
    with Path(path).open("rb") as file:
        raw = file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("release artifact exceeds evidence size limit")
    return strict_json(raw)


def pinned_json(path, expected):
    if not path or not expected:
        raise ValueError("trusted evidence is not configured")
    value = load_json_file(path)
    if digest(canonical_json(value)) != expected:
        raise ValueError("trusted evidence digest mismatch")
    return value


def verify_assessment(settings, request):
    registry = pinned_json(settings.reviewer_registry_path, settings.reviewer_registry_sha256)
    packet_hash = request["packet"]["packet_sha256"]
    return assess_signed_annotations(request, registry=registry,
                                     expected_registry_sha256=settings.reviewer_registry_sha256,
                                     expected_packet_sha256=packet_hash)


def effective_judgments(request):
    a, b = [s["payload"] for s in request["submissions"]]
    originals = {x["unit_id"]: x["judgment"] for x in a["annotations"]}
    if "adjudication" in request:
        for decision in request["adjudication"]["payload"]["decisions"]:
            if decision["resolution"] == "resolved":
                originals[decision["unit_id"]] = decision["judgment"]
    return originals


def case_review_gate(settings, record, target, source_hash, require_grounding=False, context=None):
    try:
        if record is None or record.kind != "legal_review":
            raise ValueError("signed case review is missing")
        if not context or not all(context.get(k) for k in ("actor", "at_date", "jurisdiction", "capacity")):
            raise ValueError("production case scope requires actor, date, jurisdiction and capacity")
        signed = record.request_payload["assessment"]
        expected_scope = {"actor": context["actor"], "as_of": context["at_date"],
                          "jurisdiction": context["jurisdiction"], "capacity": context["capacity"]}
        if any(u["family"]["context"] != expected_scope for u in signed["packet"]["units"]):
            raise ValueError("signed case context does not match evaluation scope")
        if record.request_payload["target_sha256"] != target:
            raise ValueError("case review is stale for this source/model/context")
        result = verify_assessment(settings, signed)  # Recheck current registry/revocation/time.
        if result["status"] != "annotations_collected":
            raise ValueError("independent case disagreements remain unresolved")
        documents = [d for u in signed["packet"]["units"] for d in u["family"]["documents"] if d["kind"] == "normative"]
        if not documents or any(d["sha256"] != source_hash for d in documents):
            raise ValueError("signed case packet does not match the evaluated source")
        payloads = [s["payload"] for s in signed["submissions"]]
        if "adjudication" in signed:
            payloads.append(signed["adjudication"]["payload"])
        if any(p.get("target_sha256") != target for p in payloads):
            raise ValueError("each independent signature must bind the exact evaluation target")
        for judgment in effective_judgments(signed).values():
            labels = judgment["labels"]
            if any(labels[d] != "supported" for d in ("applicability", "faithfulness", "conditions_exceptions", "response_behavior")):
                raise ValueError("independent case judgment is adverse or uncertain")
            if labels["grounding"] not in ({"supported"} if require_grounding else {"supported", "not_applicable"}):
                raise ValueError("case factual grounding is unconfirmed")
            if labels["harmful_assistance"] not in {"unsupported", "not_applicable"}:
                raise ValueError("independent case review identifies unresolved harmful assistance")
        return {"passed": True, "assessment_id": record.id, "assessment_sha256": digest(canonical_json(signed))}
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return {"passed": False, "reason": "missing_stale_invalid_or_unfavorable_signed_case_review"}


def deployment_release_gate(settings):
    """Validate the pinned artifact graph, current signatures, scope and thresholds."""
    try:
        manifest = pinned_json(settings.release_evidence_path, settings.release_evidence_sha256)
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "build_revision", "deployment_id", "expires_at", "approved_by", "evidence"}
                or manifest["schema"] != "vle-release-evidence/v1" or not settings.build_revision
                or manifest["build_revision"] != settings.build_revision):
            raise ValueError("release does not bind this exact build")
        expiry = datetime.fromisoformat(manifest["expires_at"])
        if expiry.tzinfo is None or expiry <= datetime.now(UTC):
            raise ValueError("release evidence expired")
        approvers = manifest["approved_by"]
        if not isinstance(approvers, list) or len(set(approvers)) < 2 or any(not isinstance(x, str) or not x.strip() for x in approvers):
            raise ValueError("two independent organizational release approvers required")
        root = Path(settings.release_evidence_path).resolve().parent
        artifacts = {}
        if set(manifest["evidence"]) != {"benchmark", "annotations", "model", "smoke", "load", "restore", "legifrance"}:
            raise ValueError("release evidence incomplete")
        for kind, ref in manifest["evidence"].items():
            if not isinstance(ref, dict) or set(ref) != {"filename", "sha256"} or not isinstance(ref["filename"], str):
                raise ValueError("invalid release artifact reference")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}\.json", ref["filename"]) or not HASH.fullmatch(ref["sha256"]):
                raise ValueError("artifact references require flat JSON filenames and SHA-256")
            file = (root / ref["filename"]).resolve()
            if file.parent != root:
                raise ValueError("release artifact escapes trusted directory")
            artifacts[kind] = pinned_json(file, ref["sha256"])
        signed = verify_assessment(settings, artifacts["annotations"])
        benchmark = artifacts["benchmark"]
        if (signed["status"] != "annotations_collected" or benchmark.get("schema") != "vle-production-benchmark/v1"
                or benchmark.get("split") != "test" or benchmark.get("build_revision") != settings.build_revision
                or benchmark.get("corpus_sha256") != artifacts["annotations"]["packet"]["packet_sha256"]
                or benchmark.get("independent_assessment_sha256") != digest(canonical_json(artifacts["annotations"]))):
            raise ValueError("held-out benchmark does not bind independent signed gold evidence")
        n = benchmark["case_count"]
        if (type(n) is not int or n < settings.release_min_cases
                or n != len(artifacts["annotations"]["packet"]["units"])
                or any(u["split"] != "test" for u in artifacts["annotations"]["packet"]["units"])):
            raise ValueError("insufficient independently annotated held-out cases")
        for name, minimum in (("extraction_f1", settings.release_min_extraction_f1), ("compliance_accuracy", settings.release_min_compliance_accuracy)):
            score = benchmark[name]
            if type(score) not in {float, int} or not minimum <= score <= 1:
                raise ValueError("held-out quality threshold not met")
        model = artifacts["model"]
        expected_endpoint = settings.model_endpoint or ("http://localhost:11434/api/chat" if settings.model_provider == "ollama" else "http://localhost:8000/v1/chat/completions")
        if (model.get("schema") != "vle-live-result/v1" or model.get("generation_calls") != 1
                or model.get("candidate") is None or model.get("status") != "review_required"
                or model.get("provider") != settings.model_provider or model.get("model") != settings.model_name
                or model.get("endpoint") != expected_endpoint
                or not HASH.fullmatch(model.get("response_content_sha256", ""))):
            raise ValueError("real inference evidence missing or wrong configured model")
        parse_family(model["review_request"])
        candidate = model["candidate"]
        if (candidate["document"] not in model["review_request"]["documents"]
                or candidate["behavior"] != model["review_request"]["behavior"]
                or not HASH.fullmatch(model.get("prompt_sha256", ""))):
            raise ValueError("inference evidence has an invalid candidate/mapping binding")
        for kind in ("smoke", "load", "restore", "legifrance"):
            report = artifacts[kind]
            if (report.get("schema") != "vle-operational-evidence/v1" or report.get("kind") != kind
                    or report.get("passed") is not True or report.get("deployment_id") != manifest["deployment_id"]
                    or report.get("build_revision") != settings.build_revision):
                raise ValueError("operational release evidence missing")
        if (artifacts["smoke"].get("audit_verified") is not True
                or not HASH.fullmatch(artifacts["restore"].get("snapshot_sha256", ""))
                or any(artifacts["legifrance"].get("corpora", {}).get(kind, 0) < 1 for kind in ("ip", "competition"))):
            raise ValueError("smoke/restore/live-source evidence is incomplete")
        load = artifacts["load"]
        if (load.get("coverage") != "authenticated_evaluation_with_audit_writes"
                or type(load.get("requests")) is not int or load["requests"] < settings.release_min_load_requests
                or load.get("errors") != 0 or type(load.get("p95_ms")) not in {int, float}
                or not 0 <= load["p95_ms"] <= settings.release_max_p95_ms):
            raise ValueError("representative persisted evaluation load/SLO evidence is missing")
        return {"passed": True, "release_sha256": settings.release_evidence_sha256,
                "deployment_id": manifest["deployment_id"], "expires_at": manifest["expires_at"],
                "basis": "externally_pinned_organizational_release_attestation", "legal_certification": False}
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return {"passed": False, "reason": "missing_expired_invalid_or_insufficient_release_evidence", "legal_certification": False}
