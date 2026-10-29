"""DDL-only features absent from NormInterpretation. No external engine claim."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..normative_model import Modality

CODES = {'O': Modality.OBLIGATION, 'P': Modality.PERMISSION, 'F': Modality.PROHIBITION}
ARROWS = {'strict': '->', 'defeasible': '=>', 'defeater': '~>'}
ID = re.compile(r'[A-Za-z_][A-Za-z0-9_-]{0,63}\Z')


@dataclass(frozen=True)
class ModalLiteral:
    modality: Modality
    atom: str
    negated: bool = False  # outer modal negation, distinct from ~atom

    def __post_init__(self):
        if (self.modality not in CODES.values() or not isinstance(self.atom, str) or not self.atom.strip()
                or len(self.atom) > 1024 or type(self.negated) is not bool):
            raise ValueError('invalid modal literal')

    @property
    def code(self):
        return next(c for c, m in CODES.items() if m == self.modality)

    def to_list(self):
        return [self.code, self.atom, self.negated]


def literal(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3 or raw[0] not in CODES:
        raise ValueError('modal literal requires [O|P|F, atom, outer_negated]')
    return ModalLiteral(CODES[raw[0]], raw[1], raw[2])


@dataclass(frozen=True)
class DDLRule:
    id: str
    kind: str
    body: tuple[str, ...]
    consequent: ModalLiteral
    exceptions: tuple[str, ...] = ()
    compensation: tuple[ModalLiteral, ...] = ()

    def __post_init__(self):
        if not isinstance(self.id, str) or not ID.fullmatch(self.id) or self.kind not in ARROWS:
            raise ValueError('invalid DDL rule id/kind')
        if any(not isinstance(x, str) or not x.strip() or len(x) > 512 for x in (*self.body, *self.exceptions)):
            raise ValueError('invalid DDL predicate')
        if len(self.body) + len(self.exceptions) > 64 or len(self.compensation) > 8:
            raise ValueError('DDL predicate/compensation limit')

    def to_dict(self):
        return {'id': self.id, 'kind': self.kind, 'body': list(self.body),
                'consequent': self.consequent.to_list(), 'exceptions': list(self.exceptions),
                'compensation': [c.to_list() for c in self.compensation]}


def parse_document(raw):
    if (not isinstance(raw, dict) or set(raw) != {'rules', 'superiority'}
            or not isinstance(raw['rules'], list) or len(raw['rules']) > 128
            or not isinstance(raw['superiority'], list) or len(raw['superiority']) > 256):
        raise ValueError('invalid DDL document')
    rules = []
    for r in raw['rules']:
        if (not isinstance(r, dict) or set(r) != {'id', 'kind', 'body', 'consequent', 'exceptions', 'compensation'}
                or not isinstance(r['body'], list) or not isinstance(r['exceptions'], list)
                or not isinstance(r['compensation'], list)):
            raise ValueError('invalid DDL rule fields')
        rules.append(DDLRule(r['id'], r['kind'], tuple(r['body']), literal(r['consequent']),
                             tuple(r['exceptions']), tuple(literal(c) for c in r['compensation'])))
    ids = {r.id for r in rules}
    if len(ids) != len(rules):
        raise ValueError('duplicate DDL rule id')
    for pair in raw['superiority']:
        if not isinstance(pair, list) or len(pair) != 2 or any(id not in ids for id in pair) or pair[0] == pair[1]:
            raise ValueError('invalid superiority edge')
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_edges_from(raw['superiority'])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('cyclic superiority relation')
    return rules, sorted(set(tuple(p) for p in raw['superiority']))


def document(rules, superiority=()):
    return {'rules': [r.to_dict() for r in sorted(rules, key=lambda r: r.id)],
            'superiority': sorted(map(list, superiority))}
