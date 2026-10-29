"""Conservative lexical snippets with exact spans, never accepted interpretations."""

from __future__ import annotations

import re

from ..audit_bundle import digest

CUES = re.compile(r"\b(must\s+not|shall\s+not|may\s+not|must|shall|may|should|unless|except|if|when)\b", re.I)


def segment(text, source_id):
    if not isinstance(text, str) or not text.strip() or len(text) > 32000 or not source_id:
        raise ValueError("source requires id and 1-32000 characters")
    matches = list(CUES.finditer(text))
    pieces, start, kind = [], 0, "context"
    for m in matches:
        if m.start() > start:
            pieces.append((kind, start, m.start()))
        cue = m.group().lower()
        if cue in {"if", "when", "unless", "except"}:
            start, kind = m.start(), "exception" if cue in {"unless", "except"} else "condition"
        else:
            pieces.append(("modality", m.start(), m.end()))
            start, kind = m.end(), "consequent"
    if start < len(text):
        pieces.append((kind, start, len(text)))
    return [
        {
            "kind": kind,
            "source_id": source_id,
            "start": start,
            "end": end,
            "quote": text[start:end],
            "document_sha256": digest(text.encode()),
            "review_status": "unreviewed",
            "scope_verified": False,
        }
        for kind, start, end in pieces
    ]
