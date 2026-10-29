"""Conservative lexical alignment. No embeddings, synonyms or guessed lemmas."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from ..normative_model import Predicate


def normalize(name):
    return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", name).casefold()).strip()


def align(norms, *, explicit=None):
    explicit = explicit or {}
    if not isinstance(explicit, dict) or set(explicit) - {"maps", "reviewer", "rationale"}:
        raise ValueError("invalid alignment map")
    maps = explicit.get("maps", {})
    if maps and (not all(isinstance(explicit.get(k), str) and explicit[k].strip() for k in ("reviewer", "rationale"))):
        raise ValueError("explicit alignment requires reviewer and rationale")
    names = {n.id: {p.name for p in (*n.conditions, *n.exceptions)} for n in norms}
    reference = min(norms, key=lambda n: (-len(names[n.id]), n.id))
    vocabulary = names[reference.id]
    if (
        not isinstance(maps, dict)
        or set(maps) - set(names)
        or any(
            not isinstance(m, dict)
            or set(m) - names[id]
            or any(not isinstance(target, str) or not target.strip() for target in m.values())
            for id, m in maps.items()
        )
    ):
        raise ValueError("alignment references unknown candidate/atom")
    aligned, records, unresolved = [], [], []
    for norm in norms:
        mapping = {}
        for name in sorted(names[norm.id]):
            proposed = [v for v in vocabulary if normalize(v) == normalize(name)]
            if name in maps.get(norm.id, {}):
                target, method = maps[norm.id][name], "reviewer_map"
            elif name in vocabulary:
                target, method = name, "exact"
            elif len(proposed) == 1:
                target, method = proposed[0], "lexical_normalization"
            else:
                unresolved.append({"candidate": norm.id, "atom": name, "reason": "unaligned_or_ambiguous"})
                continue
            mapping[name] = target
            records.append({"candidate": norm.id, "atom": name, "target": target, "method": method})
        if len(set(mapping.values())) != len(mapping):
            unresolved.append({"candidate": norm.id, "atom": "*", "reason": "many_to_one_collision"})
        aligned.append(
            replace(
                norm,
                conditions=tuple(Predicate(mapping.get(p.name, p.name), p.negated) for p in norm.conditions),
                exceptions=tuple(Predicate(mapping.get(p.name, p.name), p.negated) for p in norm.exceptions),
            )
        )
    return aligned, {
        "reference_candidate": reference.id,
        "maps": records,
        "unresolved": unresolved,
        "reviewer_attestation": explicit or None,
        "identity_verified": False,
    }
