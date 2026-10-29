"""Strict parser for local DDL/1 text; rejects unknown constructs instead of dropping them."""
from __future__ import annotations

import json

from ..norm_codec import decode_norm
from ..normative_model import InterpretationSet
from .model import ARROWS, ID, DDLRule, document, literal, parse_document


def import_document(text):
    if not isinstance(text, str) or len(text) > 1024 * 1024:
        raise ValueError('DDL text exceeds 1 MiB')
    lines = text.splitlines()
    if not lines or lines[0] != 'DDL/1' or len(lines) > 386:
        raise ValueError('expected bounded local DDL/1 text')
    rules, superiority = [], []
    decoder = json.JSONDecoder()
    for line in lines[1:]:
        if line.startswith('superiority '):
            pair = line.removeprefix('superiority ').split(' > ')
            if len(pair) != 2:
                raise ValueError('invalid superiority syntax')
            superiority.append(pair)
            continue
        id, sep, rest = line.partition(': ')
        if not sep or not ID.fullmatch(id):
            raise ValueError('invalid rule syntax')
        body, index = decoder.raw_decode(rest)
        rest = rest[index:].strip()
        arrow, sep, rest = rest.partition(' ')
        kinds = {v: k for k, v in ARROWS.items()}
        if not sep or arrow not in kinds:
            raise ValueError('unsupported DDL arrow')
        negated = rest.startswith('~')
        if negated:
            rest = rest[1:]
        if len(rest) < 3 or rest[0] not in 'OPF' or rest[1] != '(':
            raise ValueError('unsupported DDL modality')
        code = rest[0]
        target, index = decoder.raw_decode(rest[2:])
        rest = rest[2 + index:]
        if not rest.startswith(') UNLESS '):
            raise ValueError('expected UNLESS clause')
        exceptions, index = decoder.raw_decode(rest[9:])
        rest = rest[9 + index:]
        if not rest.startswith(' COMPENSATE '):
            raise ValueError('expected COMPENSATE clause')
        compensation = json.loads(rest[12:])
        if not isinstance(body, list) or not isinstance(exceptions, list) or not isinstance(compensation, list):
            raise ValueError('body, exception and compensation must be arrays')
        rules.append(DDLRule(id, kinds[arrow], tuple(body), literal([code, target, negated]),
                             tuple(exceptions), tuple(literal(c) for c in compensation)))
    raw = document(rules, superiority)
    parse_document(raw)
    return {'document': raw, 'lossy_fields': []}


def import_exported(exported):
    parsed = import_document(exported['text'])
    sidecar = exported.get('metadata_sidecar')
    if sidecar is None:
        # DDL supports semantics absent in typed norms. Do not silently project
        # strictness, outer modal negation, compensation or superiority away.
        return {**parsed, 'typed_norms': [], 'lossy_fields': ['typed_norm_projection:not_requested_no_automatic_mapping']}
    norms = [decode_norm(n) for n in sidecar['norms']]
    from .export import export_norms
    family = sidecar.get('interpretation_set')
    value = InterpretationSet(family['source_norm_id'], tuple(norms), family.get('selected_id'),
                              family.get('selection_rationale')) if family else norms
    expected = export_norms(value)
    if expected['text'] != exported['text']:
        raise ValueError('sidecar does not match DDL text')
    # Restoration preserves bytes/metadata, not DDL reasoning support.
    return {**parsed, 'typed_norms': [n.to_dict() for n in norms], 'interpretation_set': family,
            'lossy_fields': expected['lossy_fields'], 'sidecar_restoration_only': True}
