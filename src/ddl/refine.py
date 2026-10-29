"""Non-overwriting refinement proposals with componentwise check regression guards."""
from __future__ import annotations

from ..audit_bundle import canonical_json, digest
from ..references import ReferenceResolver
from .model import parse_document
from .segment import segment


def checks(raw, text, *, known_predicates=None, references=(), reference_index=None):
    if (not isinstance(references, (list, tuple)) or len(references) > 64
            or any(not isinstance(r, str) for r in references)
            or known_predicates is not None and (not isinstance(known_predicates, list) or len(known_predicates) > 512)):
        raise ValueError('bounded references/predicate inventory required')
    try:
        rules, _ = parse_document(raw)
    except (ValueError, TypeError, KeyError):
        return {'syntax_errors': 1, 'predicate_errors': None, 'missing_exceptions': None, 'dangling_references': None}
    predicates = {a for r in rules for a in (*r.body, *r.exceptions)}
    exception_cues = [s for s in segment(text, 'inline') if s['kind'] == 'exception']
    # Presence-only lexical heuristic: no claim it checks exception scope.
    return {'syntax_errors': 0, 'predicate_errors': None if known_predicates is None else len(predicates - set(known_predicates)),
            'missing_exceptions': len(exception_cues) if not any(r.exceptions for r in rules) else 0,
            'dangling_references': sum(ReferenceResolver(reference_index or {}).resolve(r).status != 'resolved' for r in references)}


def propose_refinement(before, after, text, **options):
    old, new = checks(before, text, **options), checks(after, text, **options)
    comparable = all((a is None and b is None) or (a is not None and b is not None)
                     for a, b in zip(old.values(), new.values(), strict=True))
    nonworsening = comparable and all(new[k] <= old[k] for k in old if old[k] is not None)
    return {'before_sha256': digest(canonical_json(before)), 'after_sha256': digest(canonical_json(after)),
            'checks_before': old, 'checks_after': new, 'nonworsening': nonworsening,
            'eligible_for_review': nonworsening and new['syntax_errors'] == 0,
            'accepted': False, 'review_status': 'unreviewed',
            'diff': [{'path': '/' + key, 'before': before.get(key), 'after': after.get(key)}
                     for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]}
