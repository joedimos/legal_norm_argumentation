"""Strict finite UTF-8 JSON shared by provider, CLI and evidence boundaries."""

from __future__ import annotations

import json
import math

MAX_DEPTH = 64


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite JSON")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value

    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8-sig")
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float)
        pending = [(result, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > MAX_DEPTH:
                raise ValueError("JSON nesting bound")
            if isinstance(item, dict):
                for key, value in item.items():
                    key.encode("utf-8")
                    pending.append((value, depth + 1))
            elif isinstance(item, list):
                pending.extend((v, depth + 1) for v in item)
            elif isinstance(item, str):
                item.encode("utf-8")
        return result
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid JSON encoding or nesting") from exc
