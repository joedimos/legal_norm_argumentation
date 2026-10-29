"""Lossless JSON codec for the existing typed norm (no new semantic defaults)."""

from __future__ import annotations

from dataclasses import fields

from .normative_model import Modality, NormInterpretation, Predicate, SourceAuthority, SourceKind, SourceProvenance
from .phase4 import integer


def decode_norm(raw):
    if not isinstance(raw, dict):
        raise ValueError("norm must be an object")
    data = dict(raw)
    canonical = data.pop("canonical", None)
    fingerprint = data.pop("semantic_hash", None)
    if set(data) - {f.name for f in fields(NormInterpretation)}:
        raise ValueError("unsupported typed norm fields")
    for key in ("id", "subject", "action"):
        if not isinstance(data.get(key), str) or not data[key].strip() or len(data[key]) > 512:
            raise ValueError("invalid norm identifier/target")
    data.setdefault("source_norm_id", data["id"])
    if not isinstance(data["source_norm_id"], str) or not data["source_norm_id"]:
        raise ValueError("invalid source norm id")
    data["modality"] = Modality(data["modality"])
    for key in ("conditions", "exceptions"):
        values = data.get(key, [])
        if not isinstance(values, (list, tuple)) or len(values) > 32:
            raise ValueError("predicate lists require 0-32 entries")
        if any(
            not isinstance(p, dict)
            or set(p) - {"name", "negated"}
            or not isinstance(p.get("name"), str)
            or len(p["name"]) > 512
            or type(p.get("negated", False)) is not bool
            for p in values
        ):
            raise ValueError("invalid predicate")
        data[key] = tuple(Predicate(**p) for p in values)
    authority = dict(data.get("authority", {}))
    if "kind" in authority:
        authority["kind"] = SourceKind(authority["kind"])
    data["authority"] = SourceAuthority(**authority)
    for key in ("references", "assumptions"):
        values = data.get(key, [])
        if not isinstance(values, (list, tuple)) or len(values) > 64 or any(not isinstance(x, str) for x in values):
            raise ValueError("invalid references/assumptions")
        data[key] = tuple(values)
    for key in ("object", "deadline"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValueError(f"invalid {key}")
    if data.get("provenance") is not None:
        p = dict(data["provenance"])
        if not isinstance(p.get("span"), (list, tuple)) or len(p["span"]) != 2:
            raise ValueError("invalid source span")
        start, end = p["span"]
        integer(start, 0, 1000000, "span start")
        integer(end, start, 1000000, "span end")
        p["span"] = tuple(p["span"])
        data["provenance"] = SourceProvenance(**p)
    norm = NormInterpretation(**data)
    if canonical is not None and canonical != norm.canonical():
        raise ValueError("canonical text mismatch")
    if fingerprint is not None and fingerprint != norm.semantic_hash():
        raise ValueError("semantic hash mismatch")
    return norm
