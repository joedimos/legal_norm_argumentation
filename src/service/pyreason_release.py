"""Backend-specific organizational evidence; never creates legal certification.

A pinned manifest is a custodian's attestation, not proof of experimental truth.
Native synthetic qualification cannot replace independent held-out IP judgments.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from src.audit_bundle import canonical_json, digest
from src.pyreason_runtime import installed_identity

from .ip_release import SCOPE
from .release_policy import deployment_release_gate, pinned_json, verify_assessment


def artifact(root, ref):
    if (not isinstance(ref, dict) or set(ref) != {"filename", "sha256"}
            or not isinstance(ref["filename"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}\.json", ref["filename"])
            or not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("invalid evidence reference")
    path = (root / ref["filename"]).resolve()
    if path.parent != root:
        raise ValueError("evidence reference escapes trusted directory")
    return pinned_json(path, ref["sha256"])


def pyreason_release_gate(settings):
    result = {"name": "pyreason", "passed": False, "release_qualified": False, "legal_certification": False}
    try:
        if not (settings.pyreason_enabled and settings.release_profile == "ip-supervised-v1"
                and settings.release_enforcement == "strict" and deployment_release_gate(settings)["passed"]):
            raise ValueError("strict independently validated IP release required")
        m = pinned_json(settings.pyreason_release_path, settings.pyreason_release_sha256)
        if (set(m) != {"schema", "scope", "build_revision", "deployment_id", "expires_at", "approved_by",
                       "worker_mode", "release_sha256", "native"}
                or m["schema"] != "vle-pyreason-release/v1" or m["scope"] != SCOPE
                or m["build_revision"] != settings.build_revision or m["worker_mode"] != settings.pyreason_worker_mode
                or m["release_sha256"] != settings.release_evidence_sha256):
            raise ValueError("backend release binding differs")
        expiry = datetime.fromisoformat(m["expires_at"])
        if expiry.tzinfo is None or expiry <= datetime.now(UTC):
            raise ValueError("backend evidence expired")
        approvers = m["approved_by"]
        if not isinstance(approvers, list) or len(set(approvers)) < 2 or any(not isinstance(x, str) or not x.strip() for x in approvers):
            raise ValueError("two independent custodial approvals required")
        native = artifact(Path(settings.pyreason_release_path).resolve().parent, m["native"])
        identity = installed_identity()  # No import/JIT or provider calls in a policy check.
        if (native.get("schema") != "vle-pyreason-qualification/v2" or native.get("status") != "passed"
                or native.get("provider") != "native" or native.get("runtime") != identity
                or native.get("adapter_contract_sha256") != identity["adapter_contract_sha256"]
                or native.get("build_revision") != settings.build_revision or native.get("worker_mode") != settings.pyreason_worker_mode
                or native.get("native_activation_cases", 0) < 162 or native.get("native_graph_cases", 0) < 7
                or not native.get("checks") or any(c.get("passed") is not True for c in native["checks"])
                or native.get("truncated", False) is not False):
            raise ValueError("incomplete or incompatible native evidence")
        from src.pyreason_connector import digest as program_digest
        from src.pyreason_qualification import qualification_corpus
        if native.get("corpus_sha256") != program_digest(qualification_corpus()):
            raise ValueError("different qualification corpus")
        perf = native["performance"]
        count = perf["native"]["count"]
        if (type(count) is not int or count < 32 or count > 128 or count != native["bounds"]["iterations"]
                or len(perf["native"]["samples_ms"]) != count
                or perf["norm_counts"] != [(1, 8, 32, 64)[i % 4] for i in range(count)]):
            raise ValueError("representative bounded workload is incomplete")
        expected_checks = ({f"activation_{kind}_{i}" for kind in ("batch", "values") for i in range(6)}
                           | {c["id"] for c in qualification_corpus()["graph_cases"]}
                           | {"derived_fact_controls_compliance"}
                           | {f"performance_result_{i}" for i in range(count)})
        if {c["name"] for c in native["checks"]} != expected_checks or len(native["checks"]) != len(expected_checks):
            raise ValueError("native checks are incomplete or duplicated")
        release = pinned_json(settings.release_evidence_path, settings.release_evidence_sha256)
        if release["deployment_id"] != m["deployment_id"]:
            raise ValueError("deployment differs")
        root = Path(settings.release_evidence_path).resolve().parent
        benchmark = artifact(root, release["evidence"]["benchmark"])
        assessment = artifact(root, release["evidence"]["annotations"])
        # These fields bind BOTH independent signatures to the backend run and
        # raw predictions already required by the production IP benchmark.
        target = {"schema": "vle-pyreason-qualification-target/v1", "build_revision": settings.build_revision,
                  "worker_mode": settings.pyreason_worker_mode, "runtime_sha256": identity["identity_sha256"],
                  "native_sha256": m["native"]["sha256"], "predictions_sha256": benchmark["predictions_sha256"],
                  "packet_sha256": assessment["packet"]["packet_sha256"]}
        if (benchmark.get("reasoning_backend") != "pyreason" or benchmark.get("backend_target_sha256") != digest(canonical_json(target))
                or verify_assessment(settings, assessment)["status"] != "annotations_collected"):
            raise ValueError("independent held-out native benchmark required")
        payloads = [s["payload"] for s in assessment["submissions"]]
        if "adjudication" in assessment:
            payloads.append(assessment["adjudication"]["payload"])
        if any(p.get("target_sha256") != digest(canonical_json(target)) for p in payloads):
            raise ValueError("independent signatures do not bind backend qualification")
        load = artifact(root, release["evidence"]["load"])
        if (load.get("reasoning_backend") != "pyreason" or load.get("worker_mode") != settings.pyreason_worker_mode
                or load.get("backend_computations_checked") is not True):
            raise ValueError("persisted native-backend load evidence required")
        result.update(passed=True, release_qualified=True, release_sha256=settings.pyreason_release_sha256,
                      runtime_sha256=identity["identity_sha256"], expires_at=m["expires_at"],
                      basis="externally_pinned_native_and_independent_IP_evidence")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError, ImportError):
        result["reason"] = "missing_stale_incompatible_or_insufficient_native_and_independent_IP_evidence"
    return result
