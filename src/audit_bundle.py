"""Deterministic, bounded, offline-verifiable audit archives.

Hashes detect corruption. Authenticity requires a separately trusted manifest
digest (or external signature); a self-contained hash is not a signature.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from importlib.metadata import distributions
from platform import python_version

MAX_BYTES = 16 * 1024 * 1024
MAX_FILES = 32
SCHEMA = "vle-audit-bundle/v1"
SAFE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\.json\Z")


def canonical_json(value) -> bytes:
    # Validate keys/types before json.dumps: it otherwise coerces integer keys.
    def validate(item, depth=0):
        if depth > 64:
            raise ValueError("JSON nesting exceeds 64 levels")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child, depth + 1)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("artifact must contain only JSON values")
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runtime_evidence(*, revision: str | None = None) -> dict:
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase Git SHA")
    return {"python": python_version(), "code_revision": revision,
            "revision_recorded": revision is not None,
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])),
            "semantics": {"legacy": "grounded-scoped-norms-v1", "temporal": "LTLf-complete-valuations-v1"}}


def create_bundle(artifacts: Mapping[str, object], *, metadata: dict | None = None) -> tuple[bytes, str]:
    if not 1 <= len(artifacts) <= MAX_FILES - 1:
        raise ValueError("bundle requires 1-31 artifacts")
    files = {}
    for name, payload in artifacts.items():
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name == "manifest.json":
            raise ValueError("artifact filenames must be safe flat JSON names")
        files[name] = canonical_json(payload)
    manifest = {"schema": SCHEMA, "hash_algorithm": "sha256", "metadata": metadata or {},
                "entries": {name: {"sha256": digest(data), "size": len(data)} for name, data in sorted(files.items())}}
    raw_manifest = canonical_json(manifest)
    files["manifest.json"] = raw_manifest
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError("bundle exceeds 16 MiB limit")
    output = io.BytesIO()
    # ZIP_STORED is reproducible across compression library versions.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, data)
    if len(output.getvalue()) > MAX_BYTES:
        raise ValueError("archive including headers exceeds 16 MiB limit")
    return output.getvalue(), digest(raw_manifest)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _json(data):
    value = json.loads(data, object_pairs_hook=_unique_object)
    if canonical_json(value) != data:
        raise ValueError("bundle JSON must use canonical encoding")
    return value


def verify_bundle(data: bytes, *, expected_manifest_digest: str | None = None) -> dict:
    """Never extracts files; rejects unlisted members, paths, symlinks and bombs."""
    if expected_manifest_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_digest):
        raise ValueError("trusted manifest digest must be a lowercase SHA-256 digest")
    try:
        if not isinstance(data, bytes) or len(data) > MAX_BYTES:
            raise ValueError("archive exceeds size limit")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = [e.filename for e in entries]
            if not 2 <= len(entries) <= MAX_FILES or len(set(names)) != len(names):
                raise ValueError("invalid member count or duplicate ZIP member")
            if sum(e.file_size for e in entries) > MAX_BYTES:
                raise ValueError("uncompressed size limit exceeded")
            for entry in entries:
                if (not SAFE_NAME.fullmatch(entry.filename) or entry.flag_bits & 1
                        or entry.compress_type != zipfile.ZIP_STORED
                        or (entry.external_attr >> 16) & 0o170000 != 0o100000):
                    raise ValueError("unsafe ZIP member")
            if "manifest.json" not in names:
                raise ValueError("missing manifest")
            raw = archive.read("manifest.json")
            manifest = _json(raw)
            if not isinstance(manifest, dict) or set(manifest) != {"schema", "hash_algorithm", "metadata", "entries"}:
                raise ValueError("invalid manifest structure")
            if manifest["schema"] != SCHEMA or manifest["hash_algorithm"] != "sha256":
                raise ValueError("unsupported manifest schema or hash algorithm")
            if not isinstance(manifest["entries"], dict) or not isinstance(manifest["metadata"], dict):
                raise ValueError("invalid manifest fields")
            if set(names) != {"manifest.json", *manifest["entries"]}:
                raise ValueError("manifest membership mismatch")
            for name, expected in manifest["entries"].items():
                if (not isinstance(expected, dict) or set(expected) != {"sha256", "size"}
                        or type(expected["size"]) is not int or expected["size"] < 0
                        or not isinstance(expected["sha256"], str)
                        or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
                    raise ValueError("invalid manifest entry")
                payload = archive.read(name)
                if expected != {"sha256": digest(payload), "size": len(payload)}:
                    raise ValueError("artifact digest or length mismatch")
                _json(payload)
            actual = digest(raw)
            if expected_manifest_digest is not None and actual != expected_manifest_digest:
                raise ValueError("trusted manifest digest mismatch")
            return {"valid": True, "manifest_digest": actual, "artifact_count": len(names) - 1,
                    "trusted_digest_matched": expected_manifest_digest is not None,
                    "signature_verified": False, "legal_certification": False}
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, zipfile.BadZipFile, RecursionError) as exc:
        return {"valid": False, "reason": str(exc), "trusted_digest_matched": False,
                "signature_verified": False, "legal_certification": False}
