"""Synthetic, unreviewed structured inference contracts; offline oracle receipts."""
import copy
import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

from src.compliance import evaluate
from src.norm_codec import decode_norm
from src.normative_model import FactContext
from src.pyreason_connector import BACKEND_VERSION, digest
from src.pyreason_graph import LIMITS, infer_graph, normalize, reference
from src.pyreason_program import execute_program
from tools.fixtures import pyreason as fake

ROOT = Path(__file__).resolve().parents[1]


def graph():
    return json.loads((ROOT / 'examples/pyreason-graph.json').read_text())


@pytest.fixture
def offline_engine(monkeypatch):
    def execute(cmd, **kwargs):
        program = json.loads(Path(cmd[-2]).read_text())
        answers = reference(program) if program.get('mode') == 'graph_temporal_v1' else execute_program(program, fake)
        Path(cmd[-1]).write_text(json.dumps({'status': 'observed', 'backend_version': BACKEND_VERSION,
                                          'input_sha256': digest(program), 'annotations': answers}))
    monkeypatch.setattr(subprocess, 'run', execute)


def test_temporal_interval_and_relational_corpus_against_finite_oracle(offline_engine):
    corpus = json.loads((ROOT / 'benchmarks/pyreason-native-v1.json').read_text())
    assert corpus['kind'] == 'synthetic' and corpus['review_status'] == 'unreviewed'
    for case in corpus['graph_cases']:
        assert case['kind'] == 'synthetic' and case['review_status'] == 'unreviewed'
        validate(case['program'], json.loads((ROOT/'schemas/pyreason-graph.schema.json').read_text()))
        normalized, _, _ = normalize(case['program'])
        assert reference(normalized) == case['expected']
        result = infer_graph(case['program'])
        assert result['status'] == 'computed', result
        assert result['answers'] == case['expected']
        validate(result, json.loads((ROOT / 'schemas/pyreason-inference.schema.json').read_text()))


def test_derived_fact_changes_normative_result_and_conflicting_observation_blocks(offline_engine):
    rule = decode_norm(json.loads((ROOT / 'examples/pyreason-normative.json').read_text())['norms'][0])
    context = FactContext({'protected': True}, observed_actions=('reproduce',))
    assert evaluate([rule], context)['status'] == 'indeterminate'
    enriched = evaluate([rule], context, backend='pyreason', pyreason_program=graph())
    assert enriched['status'] == 'compliant'
    assert enriched['reasoning']['backend']['fact_inference']['derived_facts'] == {'licensed': True}
    contradictory = evaluate([rule], FactContext({'protected': True, 'licensed': False}), backend='pyreason', pyreason_program=graph())
    assert contradictory['status'] == 'indeterminate'
    assert contradictory['reasoning']['backend']['reason'] == 'derived_observation_conflict'
    assert contradictory['reasoning']['backend']['activations'] == {}
    with pytest.raises(ValueError):
        evaluate([rule], context, pyreason_program=graph())
    with pytest.raises(ValueError):
        evaluate([rule], context, pyreason_worker_mode='unknown')


@pytest.mark.parametrize('fault', ['missing', 'disagreement', 'soft', 'receipt', 'timeout'])
def test_invalid_native_answers_never_become_facts(monkeypatch, fault):
    def run(cmd, **kwargs):
        if fault == 'timeout':
            raise subprocess.TimeoutExpired('private', 1)
        program = json.loads(Path(cmd[-2]).read_text())
        answers = reference(program)
        if fault == 'missing':
            answers.pop('effective')
        elif fault in {'disagreement', 'soft'}:
            answers['effective'] = [0, 0] if fault == 'disagreement' else [.5, 1]
        Path(cmd[-1]).write_text(json.dumps({'status': 'observed', 'backend_version': BACKEND_VERSION,
                      'input_sha256': '0'*64 if fault == 'receipt' else digest(program), 'annotations': answers}))
    monkeypatch.setattr(subprocess, 'run', run)
    result = infer_graph(graph())
    assert result['status'] == 'unknown' and result['derived_facts'] == {} and result['answers'] == {}
    assert not result['decision_authorized']


def test_soft_observations_remain_uncertain_and_contradictions_fail_closed(offline_engine):
    raw = graph()
    for fact in raw['facts']:
        fact['interval'] = [.6, .9]
    for rule in raw['rules']:
        rule['interval'] = [.6, .9]
        for body in rule['body']:
            body['interval'] = [.6, 1]
    result = infer_graph(raw)
    assert result['status'] == 'computed'
    assert result['derived_facts'] == {} and result['uncertain_bindings'] == ['licensed']
    raw['facts'].append({**copy.deepcopy(raw['facts'][0]), 'interval': [0, 0]})
    result = infer_graph(raw)
    assert result['status'] == 'unknown' and result['reason'] == 'inconsistent_interval_constraints'


@pytest.mark.parametrize('change', [
    {'schema': 'invalid'}, {'review_status': 'approved'}, {'entities': []},
    {'predicates': {'x': 3}}, {'horizon': True}, {'bindings': {'x': 'missing'}}, {'queries': []},
])
def test_unsafe_or_incomplete_graph_requests_never_run(change, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('invalid program reached native engine'))
    assert infer_graph({**graph(), **change})['status'] == 'unknown'


def test_variable_safety_soft_to_crisp_and_unknown_guards_are_rejected():
    for alteration in ('constant', 'free_variable', 'unknown', 'soft_to_crisp', 'duplicate'):
        raw = graph()
        if alteration == 'constant':
            raw['rules'][0]['head']['arguments'][0] = 'operator'
        elif alteration == 'free_variable':
            raw['rules'][0]['head']['arguments'][0] = '$free'
        elif alteration == 'unknown':
            raw['rules'][0]['body'][0]['interval'] = [0, 1]
        elif alteration == 'soft_to_crisp':
            raw['rules'][0]['body'][0]['interval'] = [.5, 1]
        else:
            raw['rules'].append(copy.deepcopy(raw['rules'][0]))
        assert infer_graph(raw)['status'] == 'unknown'


def test_bounds_and_schema_prevent_partial_positive_output(offline_engine, monkeypatch):
    raw = graph()
    raw['facts'] *= 257
    result = infer_graph(raw)
    assert result['status'] == 'unknown' and result['bound_hit']
    monkeypatch.setitem(LIMITS, 'max_groundings', 1)
    result = infer_graph(graph())
    assert result['status'] == 'unknown' and result['bound_hit']
    assert result['bounds']['max_groundings'] == 1
    monkeypatch.setitem(LIMITS, 'max_groundings', 8192)
    result = infer_graph(graph())
    result['status'] = 'unknown'
    with pytest.raises(ValidationError):
        validate(result, json.loads((ROOT / 'schemas/pyreason-inference.schema.json').read_text()))


def test_cli_and_bundle_carry_graph_rules_and_trace(offline_engine, monkeypatch, tmp_path, capsys):
    import sys

    from src.audit_bundle import verify_bundle
    from src.cli import main
    bundle = tmp_path / 'native.zip'
    monkeypatch.setattr(sys, 'argv', ['vle-legato', 'pyreason-infer', '--input', str(ROOT/'examples/pyreason-graph.json'), '--bundle', str(bundle)])
    with pytest.raises(SystemExit) as code:
        main()
    assert code.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'computed' and report['review_status'] == 'unreviewed'
    assert verify_bundle(bundle.read_bytes())['valid']


def test_native_compiler_uses_only_normalized_atoms_and_checks_temporal_trace():
    """Recorded API shape; the separate native job checks actual semantics."""
    from types import SimpleNamespace

    from src.pyreason_graph import execute_native, key
    raw = graph()
    program,_,_ = normalize(raw)
    expected = reference(program)
    trace = {}
    for q in program['queries']:
        predicate,args = key(q['atom'])
        component = args[0] if len(args)==1 else args
        trace.setdefault(q['at'],{}).setdefault(component,{})[predicate]=expected[q['id']]
    calls=[]
    pr=SimpleNamespace(reset=lambda:None,reset_rules=lambda:None,reset_settings=lambda:None,settings=SimpleNamespace(),
            load_graph=lambda x:None,add_fact=lambda x:calls.append(x),Fact=lambda text,**kw:text,
            add_rule=lambda x:calls.append(x),Rule=lambda text,**kw:text,reason=lambda **kw:SimpleNamespace(get_dict=lambda:trace))
    assert execute_native(program,pr)==expected
    assert all('operator' not in x and 'license' not in x for x in calls)
    trace[99]={}
    with pytest.raises(ValueError):
        execute_native(program,pr)


def test_reference_update_and_time_caps_remain_unknown(offline_engine, monkeypatch):
    import src.pyreason_graph as module
    monkeypatch.setitem(LIMITS,'max_updates',0)
    result=infer_graph(graph())
    assert result['status']=='unknown' and result['bound_hit']
    monkeypatch.setitem(LIMITS,'max_updates',10000)
    normalized,_,_=normalize(graph())
    with pytest.raises(module.ResourceBound):
        reference(normalized,deadline=0)
