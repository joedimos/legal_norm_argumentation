import json
from itertools import product
from pathlib import Path

import jsonschema
import pytest

from src.argumentation import OptimizedAAF
from src.aspic import analyze_aspic, analyze_norms
from src.defeasible import reason
from src.normative_model import FactContext
from src.research import norm_from_dict


def request(rules=(), premises=(), **kw):
    return {'schema': 'vle-aspic/v1', 'premises': list(premises), 'rules': list(rules), **kw}


def rule(id, conclusion, ants=(), kind='defeasible'):
    return {'id': id, 'conclusion': conclusion, 'antecedents': list(ants), 'kind': kind}


def validate(result):
    jsonschema.validate(result, json.loads(Path('schemas/aspic-result.schema.json').read_text()))
    assert result['decision_authorized'] is result['legal_certification'] is False


def test_all_512_directed_graph_embeddings():
    names = ['x', 'y', 'z']
    edges = list(product(names, repeat=2))
    for bits in product((False, True), repeat=9):
        attacks = {e for e, bit in zip(edges, bits, strict=True) if bit}
        contraries = {b: [a for a in names if (a, b) in attacks] for b in names}
        for semantics in ('grounded', 'preferred', 'stable'):
            r = analyze_aspic(request([rule(n, n) for n in names], contraries=contraries, semantics=semantics))
            validate(r)
            f = OptimizedAAF(names, attacks=attacks)
            expected = ([f.get_grounded_extension()] if semantics == 'grounded' else
                        f.get_preferred_extensions() if semantics == 'preferred' else f.get_stable_extensions())
            mapping = {a['id']: a['conclusion'] for a in r['arguments']}
            assert sorted(sorted(mapping[a] for a in e) for e in r['extensions']) == sorted(sorted(e) for e in expected)


def test_attacks_propagate_to_superarguments_and_axioms_are_protected():
    r = analyze_aspic(request([rule('r', 'q', ['p']), rule('s', 'z', ['q'], 'strict'),
                              rule('u', '~rule:r'), rule('b', '~p'), rule('v', '~q')],
                             [{'id': 'p0', 'literal': 'p', 'kind': 'ordinary'},
                              {'id': 'a0', 'literal': '~z', 'kind': 'axiom'}]))
    validate(r)
    assert {a['kind'] for a in r['attacks']} == {'undermining', 'rebutting', 'undercutting'}
    by_id = {a['id']: a for a in r['arguments']}
    assert any(by_id[a['to']]['conclusion'] == 'z' and a['kind'] == 'undercutting' for a in r['attacks'])
    assert r['rationality']['observed_postulates']['subargument_closure']
    assert not r['rationality']['preconditions']['strict_transposition']


def test_lifting_changes_defeat_and_undercutters_ignore_preferences():
    rules = [rule('r1', 'p'), rule('r2', 'q', ['p']), rule('r3', '~q')]
    prefs = [{'preferred': 'r2', 'disfavored': 'r3', 'rationale': 'test'},
             {'preferred': 'r3', 'disfavored': 'r1', 'rationale': 'test'}]
    last = analyze_aspic(request(rules, preferences=prefs, lifting='last'))
    weak = analyze_aspic(request(rules, preferences=prefs, lifting='weakest'))
    assert last['defeats'] != weak['defeats']
    undercut = analyze_aspic(request([rule('r', 'p'), rule('s', '~rule:r')], preferences=[
        {'preferred': 'r', 'disfavored': 's', 'rationale': 'test'}]))
    assert undercut['attacks'][0]['successful']


@pytest.mark.parametrize('kw', [{'max_arguments': 1}, {'max_depth': 1}, {'max_states': 1}])
def test_bounds_never_publish_partial_acceptance(kw):
    r = analyze_aspic(request([rule('a', 'p'), rule('b', 'q', ['p'])], **kw))
    validate(r)
    assert r['status'] == 'unknown' and not r['extensions'] and not r['has_extensions']


def test_cycles_and_extension_limit():
    assert analyze_aspic(request([rule('a', 'p', ['p'])]))['status'] == 'unknown'
    r = analyze_aspic(request([rule(str(i), str(i)) for i in range(17)], semantics='preferred'))
    assert r['status'] == 'unknown' and r['reason'] == 'extension_limit'


def test_invalid_orders_and_inputs():
    base = request([rule('a', 'p'), rule('b', '~p')])
    bad = [dict(base, lifting='other'), dict(base, max_states=True), dict(base, rules=base['rules']*2),
           dict(base, preferences=[{'preferred': 'a', 'disfavored': 'a', 'rationale': 'cycle'}]),
           dict(base, premises=[{'id': 'c', 'literal': 'x', 'kind': 'bogus'}]),
           dict(base, contraries={'x': 'y'})]
    for payload in bad:
        with pytest.raises(ValueError):
            analyze_aspic(payload)


def test_rationality_for_strict_transposed_consistent_system():
    rules = [rule('s', 'q', ['p'], 'strict'), rule('t', '~p', ['~q'], 'strict')]
    r = analyze_aspic(request(rules, [{'id': 'p', 'literal': 'p', 'kind': 'axiom'}]))
    assert all(r['rationality']['preconditions'].values())
    assert all(r['rationality']['observed_postulates'].values())
    # Deliberately inconsistent axioms must not receive a consistency claim.
    bad = analyze_aspic(request([], [{'id': 'p', 'literal': 'p', 'kind': 'axiom'},
                                     {'id': 'n', 'literal': '~p', 'kind': 'axiom'}]))
    assert not bad['rationality']['observed_postulates']['direct_consistency']


def test_existing_62_case_corpus_overlap_and_explicit_out_of_scope():
    corpus = json.loads(Path('benchmarks/research-v1.json').read_text())
    assert len(corpus['cases']) == 62
    checked = 0
    for case in corpus['cases']:
        if case['kind'] != 'research':
            continue  # temporal formulas, SMT queries and arbitrary preferences lack typed norm inputs
        raw = case['input']
        if any(n['start'] != 0 for n in raw['norms']) or raw.get('preferences'):
            continue  # temporal windows and caller orders are not this single-context bridge
        norms = [norm_from_dict(n['norm']) for n in raw['norms']]
        state = raw['trace'][0]
        facts = {k[5:]: v for k, v in state.items() if k.startswith('fact:') and v is not None}
        context = FactContext(facts)
        r = analyze_norms(norms, context)
        expected = reason(norms, context)
        if r['status'] != 'unknown':
            assert r['norm_labels'] == expected['argumentation']['labels']
        checked += 1
    assert checked > 0


def test_authority_priority_bridge_and_unknown_activation():
    norms = [norm_from_dict({'id': id, 'modality': m, 'subject': 's', 'action': 'a',
                            'authority': {'authority_rank': rank}})
             for id, m, rank in [('o', 'obligation', 3), ('f', 'prohibition', 2)]]
    assert analyze_norms(norms, FactContext({}))['norm_labels'] == reason(norms, FactContext({}))['argumentation']['labels']
    norms[0] = norm_from_dict({'id': 'o', 'modality': 'obligation', 'subject': 's', 'action': 'a',
                               'conditions': [{'name': 'unknown'}]})
    assert analyze_norms(norms, FactContext({}))['status'] == 'unknown'
