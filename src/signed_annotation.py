"""Attributable legal judgments under an externally pinned reviewer registry.

Signatures authenticate registered keys. Qualifications and organizational
independence remain the registry custodian's checked assertions, not crypto facts.
"""
from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

from .audit_bundle import canonical_json, digest
from .legal_annotation import _reviewer, _text, _validate_packet, assess_annotations

DOMAIN = b"vle-independent-legal-review/v1\x00"


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:
        raise ValueError("install the annotation extra: pip install -e '.[annotation]'") from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


def _hex(value, size):
    if not isinstance(value, str) or len(value) != size * 2 or value.lower() != value:
        raise ValueError("invalid lowercase hexadecimal key/signature/hash")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid hexadecimal encoding") from exc
    if len(result) != size:
        raise ValueError("incorrect key/signature/hash length")
    return result


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("signature timestamp requires an explicit UTC offset")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("signature timestamp must be UTC")
    return result


def create_reviewer_key(private_path, public_path):
    """Never print private bytes or replace an existing key."""
    Private, _, _ = _crypto()
    key = Private.generate()
    public = key.public_key().public_bytes_raw().hex()
    # Reserve the public file first; remove only a file created by this call on failure.
    with Path(public_path).open("x", encoding="utf-8") as public_file:
        try:
            fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as private_file:
                private_file.write(key.private_bytes_raw().hex() + "\n")
            public_file.write(public + "\n")
        except BaseException:
            Path(public_path).unlink()
            raise
    return {"public_key": public, "private_key_written": True}


def sign_judgment(payload, *, packet_sha256, role, private_key_file, now=None):
    if role not in {"annotator", "adjudicator"}:
        raise ValueError("invalid signing role")
    _hex(packet_sha256, 32)
    if not isinstance(payload, dict):
        raise ValueError("judgment payload must be an object")
    _reviewer(payload.get("reviewer"))
    if role == "annotator" and payload.get("packet_sha256") != packet_sha256:
        raise ValueError("submission binds a different packet")
    Private, _, _ = _crypto()
    with Path(private_key_file).open("r", encoding="utf-8") as key_file:
        raw = key_file.read(256).strip()
    key = Private.from_private_bytes(_hex(raw, 32))
    body = {"schema": "vle-signed-judgment/v1", "role": role, "packet_sha256": packet_sha256,
            "signed_at": (now or datetime.now(UTC)).isoformat(), "payload": payload}
    _timestamp(body["signed_at"])
    signature = key.sign(DOMAIN + canonical_json(body)).hex()
    return {**body, "signature": signature}


def _registry(registry, expected, now):
    _hex(expected, 32)
    if digest(canonical_json(registry)) != expected:
        raise ValueError("reviewer registry does not match separately trusted digest")
    if (not isinstance(registry, dict) or set(registry) != {"schema", "reviewers", "excluded_independence_groups"}
            or registry["schema"] != "vle-reviewer-registry/v1" or not isinstance(registry["reviewers"], list)
            or not 2 <= len(registry["reviewers"]) <= 100):
        raise ValueError("invalid reviewer registry")
    excluded = registry["excluded_independence_groups"]
    if not isinstance(excluded, list) or not 1 <= len(excluded) <= 100:
        raise ValueError("registry must declare excluded generator/developer groups")
    for group in excluded:
        _text(group, "excluded independence group", 128)
    indexed, keys = {}, set()
    fields = {"id", "public_key", "roles", "jurisdictions", "independence_group", "valid_from", "valid_until",
              "revoked", "credential_check"}
    for person in registry["reviewers"]:
        if not isinstance(person, dict) or set(person) != fields:
            raise ValueError("invalid registry reviewer")
        _text(person["id"], "registry identity", 128)
        _hex(person["public_key"], 32)
        if person["id"] in indexed or person["public_key"] in keys:
            raise ValueError("registry identities and public keys must be unique")
        for field in ("roles", "jurisdictions"):
            values = person[field]
            if (not isinstance(values, list) or not 1 <= len(values) <= 32
                    or any(not isinstance(v, str) or not v.strip() for v in values) or len(values) != len(set(values))):
                raise ValueError("invalid registry role/jurisdiction scope")
        if not set(person["roles"]) <= {"annotator", "adjudicator"}:
            raise ValueError("invalid registry role")
        _text(person["independence_group"], "independence group", 128)
        if type(person["revoked"]) is not bool:
            raise ValueError("registry revocation must be explicit")
        start, end = date.fromisoformat(person["valid_from"]), date.fromisoformat(person["valid_until"])
        if start.isoformat() != person["valid_from"] or end.isoformat() != person["valid_until"] or start > end:
            raise ValueError("invalid registry validity dates")
        check = person["credential_check"]
        if not isinstance(check, dict) or set(check) != {"checked_by", "checked_at", "evidence_sha256"}:
            raise ValueError("registry must record a custodian credential check")
        _text(check["checked_by"], "credential checker", 128)
        if _timestamp(check["checked_at"]) > now:
            raise ValueError("credential check is in the future")
        _hex(check["evidence_sha256"], 32)
        indexed[person["id"]] = person
        keys.add(person["public_key"])
    return indexed


def assess_signed_annotations(request, *, registry, expected_registry_sha256, expected_packet_sha256, now=None):
    if (not isinstance(request, dict) or request.get("schema") != "vle-signed-assessment/v1"
            or not {"schema", "packet", "submissions"} <= set(request)
            or set(request) - {"schema", "packet", "submissions", "adjudication"}):
        raise ValueError("invalid signed assessment")
    _validate_packet(request["packet"])
    _hex(expected_packet_sha256, 32)
    if request["packet"]["packet_sha256"] != expected_packet_sha256:
        raise ValueError("packet does not match separately frozen study")
    current = now or datetime.now(UTC)
    _timestamp(current.isoformat())
    indexed = _registry(registry, expected_registry_sha256, current)
    jurisdictions = {u["family"]["context"]["jurisdiction"] for u in request["packet"]["units"]}
    _, Public, InvalidSignature = _crypto()
    used_ids, groups = set(), set()

    def verify(envelope, role):
        fields = {"schema", "role", "packet_sha256", "signed_at", "payload", "signature"}
        if (not isinstance(envelope, dict) or set(envelope) != fields
                or envelope["schema"] != "vle-signed-judgment/v1" or envelope["role"] != role
                or envelope["packet_sha256"] != expected_packet_sha256):
            raise ValueError("signature envelope role/packet mismatch")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise ValueError("invalid signed payload")
        _reviewer(payload.get("reviewer"))
        person = indexed.get(payload["reviewer"]["id"])
        if (person is None or person["revoked"] or role not in person["roles"]
                or person["independence_group"] in registry["excluded_independence_groups"]):
            raise ValueError("reviewer is unknown, revoked or outside registered role")
        signed = _timestamp(envelope["signed_at"])
        if signed > current or any(not date.fromisoformat(person["valid_from"]) <= d <= date.fromisoformat(person["valid_until"])
                                   for d in (signed.date(), current.date())):
            raise ValueError("signature/key is outside validity period")
        if not jurisdictions <= set(person["jurisdictions"]):
            raise ValueError("reviewer is outside registered jurisdiction scope")
        if person["id"] in used_ids or person["independence_group"] in groups:
            raise ValueError("reviewers/adjudicator must have separate registered identities and independence groups")
        try:
            Public.from_public_bytes(_hex(person["public_key"], 32)).verify(
                _hex(envelope["signature"], 64), DOMAIN + canonical_json({k: v for k, v in envelope.items() if k != "signature"}))
        except InvalidSignature as exc:
            raise ValueError("invalid judgment signature") from exc
        used_ids.add(person["id"])
        groups.add(person["independence_group"])
        return payload

    if not isinstance(request["submissions"], list) or len(request["submissions"]) != 2:
        raise ValueError("exactly two signed independent submissions required")
    plain = {"schema": "vle-annotation-assessment/v1", "packet": request["packet"],
             "submissions": [verify(s, "annotator") for s in request["submissions"]]}
    if "adjudication" in request:
        plain["adjudication"] = verify(request["adjudication"], "adjudicator")
    result = assess_annotations(plain)
    units = {u["id"]: u for u in request["packet"]["units"]}
    judgments = [(a["unit_id"], a["judgment"]) for s in plain["submissions"] for a in s["annotations"]]
    judgments += [(d["unit_id"], d["judgment"]) for d in plain.get("adjudication", {}).get("decisions", [])]
    for unit_id, judgment in judgments:
        if "evidence_refs" not in judgment:
            raise ValueError("signed legal judgments must explicitly declare evidence_refs")
        documents = {d["id"]: d for d in units[unit_id]["family"]["documents"]}
        kinds = {documents[r["document_id"]]["kind"] for r in judgment["evidence_refs"]}
        labels = judgment["labels"]
        if (any(labels[d] in {"supported", "unsupported"} for d in ("applicability", "faithfulness", "conditions_exceptions"))
                and "normative" not in kinds):
            raise ValueError("determinate legal interpretation judgment needs normative source evidence")
        if labels["response_behavior"] in {"supported", "unsupported"} and "draft" not in kinds:
            raise ValueError("determinate behavior judgment needs candidate source evidence")
        if labels["grounding"] == "supported" and "observation" not in kinds:
            raise ValueError("supported factual grounding needs independent observation evidence")
    return {**result, "schema": "vle-signed-assessment-result/v1", "signed_request_sha256": digest(canonical_json(request)),
            "registry_sha256": expected_registry_sha256, "key_signatures_verified": True,
            "registered_credential_checks_present": True, "authentication_basis": "externally_pinned_public_key_registry",
            "independence_basis": "distinct_custodian_registered_groups_and_reviewer_attestations",
            "verified_at": current.isoformat(), "signed_evidence": request}
