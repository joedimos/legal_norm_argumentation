import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from src.audit_bundle import create_bundle, redundancy_artifacts, verify_bundle
from src.normative_model import Modality, NormInterpretation, Predicate
from src.redundancy.crosscheck import consensus_report
from src.redundancy.generate import FakeProvider, RuntimeProvider, baseline_fake, generate_formalizations


def norm(id, **kw):
    return NormInterpretation(id, 'source', Modality.OBLIGATION, 'operator', 'notify', **kw)


def report(a, b, **kw):
    return consensus_report({'schema': 'vle-redundancy/v1', 'horizon': 1,
        'candidates': [{'norm': n.to_dict(), 'review_status': 'unreviewed'} for n in (a, b)], **kw})


def validate(result, schema='redundancy-report'):
    jsonschema.validate(json.loads(json.dumps(result)), json.loads(Path(f'schemas/{schema}.schema.json').read_text()))
    assert not result['decision_authorized'] and not result['legal_certification']


def test_agreement_is_never_selection_or_review_and_bundle_is_deterministic():
    r = report(norm('a'), norm('b'))
    validate(r)
    assert r['consensus_supported'] and r['selected_id'] is None and r['review_status'] == 'unreviewed'
    a = redundancy_artifacts({}, r)
    assert create_bundle(a) == create_bundle(dict(reversed(list(a.items()))))
    data, root = create_bundle(a)
    assert verify_bundle(data, expected_manifest_digest=root)['valid']
    with pytest.raises(ValueError):
        redundancy_artifacts({}, {})


def test_lexical_and_explicit_alignment_and_unresolved_atoms():
    a = norm('a', conditions=(Predicate('HIGH_RISK'),))
    b = norm('b', conditions=(Predicate('high-risk'),))
    assert report(a, b)['consensus_supported']
    b = replace(b, conditions=(Predicate('danger'),))
    r = report(a, b)
    assert r['status'] == 'unknown' and not r['consensus_supported']
    assert r['pairs'][0]['classification'] == 'unknown'
    r = report(a, b, alignment={'maps': {'b': {'danger': 'HIGH_RISK'}},
                              'reviewer': 'synthetic-unreviewed-example', 'rationale': 'test mapping'})
    assert r['consensus_supported'] and not r['alignment']['identity_verified']
    with pytest.raises(ValueError):
        report(a, b, alignment={'maps': {'b': {'danger': 'HIGH_RISK'}}})


def test_exception_scope_strength_and_symmetry_both_backends():
    a, b = norm('a'), norm('b', exceptions=(Predicate('waiver'),))
    for backend in ('exhaustive', 'z3'):
        if backend == 'z3':
            pytest.importorskip('z3')
        r = report(a, b, backend=backend)
        validate(r)
        p = r['pairs'][0]
        assert p['classification'] == 'strictly_stronger'
        assert p['witnesses'] and p['witnesses'][0]['horizon'] == 1
        assert report(b, a, backend=backend)['pairs'][0]['classification'] == 'strictly_weaker'


def test_modal_distinction_cannot_collapse_permission_into_obligation():
    a = norm('a')
    b = replace(norm('b'), modality=Modality.PERMISSION)
    r = report(a, b)
    assert r['pairs'][0]['classification'] == 'incomparable' and not r['consensus_supported']
    assert any(w['kind'] == 'modal_distinction' and w['trace'] for w in r['pairs'][0]['witnesses'])


@pytest.mark.parametrize('kwargs', [{'max_traces': 1}, {'horizon': 33}])
def test_bounds_and_unmodeled_metadata_fail_closed(kwargs):
    a, b = norm('a', conditions=(Predicate('risk'),)), norm('b', exceptions=(Predicate('waiver'),))
    r = report(a, b, **kwargs)
    assert r['status'] == 'unknown' and not r['consensus_supported']
    assert report(replace(norm('a'), deadline='tomorrow'), norm('b'))['status'] == 'unknown'


def test_generation_is_reproducible_unreviewed_and_records_full_provenance():
    text = 'The operator must notify.'
    a = generate_formalizations(text, 's', baseline_fake(text, 's'), candidates=3)
    b = generate_formalizations(text, 's', baseline_fake(text, 's'), candidates=3)
    validate(a, 'redundancy-generation')
    assert a == b and a['interpretation_set']['selected_id'] is None
    assert len({c['provenance']['prompt_sha256'] for c in a['candidates']}) == 3
    assert all(c['review_status'] == 'unreviewed' for c in a['candidates'])
    assert a['candidates'][0]['norm']['provenance']['quote'] == text
    broken = generate_formalizations(text, 's', FakeProvider([{}]), candidates=2)
    assert broken['status'] == 'unknown' and len(broken['failures']) == 2
    with pytest.raises(ValueError):
        baseline_fake('No norm here.', 's')
    with pytest.raises(ValueError):
        generate_formalizations(text, 's', FakeProvider([]), candidates=1)


@pytest.mark.parametrize('provider', ['ollama', 'vllm'])
def test_runtime_provider_uses_existing_transport_without_network(provider):
    seen = []
    def transport(payload, **kw):
        seen.append(payload)
        return {'done': True, 'model': 'fake', 'message': {'content': json.dumps(norm('x').to_dict())}}
    p = RuntimeProvider({'provider': provider, 'model': 'fake'}, transport=transport)
    r = generate_formalizations('The operator must notify.', 's', p, candidates=2,
                                timestamp='2026-01-01T00:00:00+00:00')
    assert len(r['candidates']) == 2
    assert seen[0] != seen[1]


def test_invalid_and_collision_inputs():
    with pytest.raises(ValueError):
        report(norm('same'), norm('same'))
    with pytest.raises(ValueError):
        report(norm('a'), norm('b'), k=True)
    a = norm('a', conditions=(Predicate('x'), Predicate('y')))
    b = norm('b', conditions=(Predicate('x'), Predicate('y')))
    r = report(a, b, alignment={'maps': {'b': {'y': 'x'}}, 'reviewer': 'synthetic', 'rationale': 'test'})
    assert r['status'] == 'unknown'


def test_pair_classification_symmetry_over_small_norm_family():
    variants = [norm('x'), norm('x', conditions=(Predicate('p'),)),
                norm('x', exceptions=(Predicate('p'),)), replace(norm('x'), modality=Modality.PERMISSION)]
    flip = {'equivalent': 'equivalent', 'strictly_stronger': 'strictly_weaker',
            'strictly_weaker': 'strictly_stronger', 'incomparable': 'incomparable', 'unknown': 'unknown'}
    for a in variants:
        for b in variants:
            left, right = replace(a, id='a'), replace(b, id='b')
            ab = report(left, right)['pairs'][0]['classification']
            ba = report(right, left)['pairs'][0]['classification']
            assert ab == flip[ba]
