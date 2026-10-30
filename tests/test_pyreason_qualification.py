"""Synthetic, unreviewed control-flow tests; real engine verified separately."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import pyreason_qualification as qualification
from src import pyreason_runtime as runtime
from src import pyreason_warmup as warmup
from src.defeasible import activation
from src.pyreason_connector import ResourceBound
from src.pyreason_graph import reference


def test_bounded_qualification_control_flow_with_explicit_standins(monkeypatch):
    monkeypatch.setattr(qualification,'run_worker',lambda *a:{'annotations':{'provider':'native','packages':{'pyreason':'3.8.0'}}})
    monkeypatch.setattr(qualification,'run_activation_backend',lambda norms,context,**kw:{'status':'computed',
                         'activations':{n.id:{'status':activation(n,context).status} for n in norms}})
    def graph(raw,**kwargs):
        from src.pyreason_graph import normalize
        program,*_=normalize(raw)
        return {'status':'computed','answers':reference(program)}
    monkeypatch.setattr(qualification,'infer_graph',graph)
    def evaluate(norms,context,**kwargs):
        return {'status':'compliant' if kwargs.get('pyreason_program') else 'indeterminate',
                'reasoning':{'backend':{'status':'computed'}}}
    monkeypatch.setattr(qualification,'evaluate',evaluate)
    result = qualification.qualify_native(iterations=4,timeout_seconds=5)
    assert result['status'] == 'passed' and result['native_activation_cases'] == 162
    assert result['performance']['norm_counts'] == [1,8,32,64]
    assert not result['decision_authorized']
    monkeypatch.setattr(qualification,'infer_graph',lambda *a,**k:{'status':'unknown','answers':{},'reason':'injected'})
    result = qualification.qualify_native(iterations=1,timeout_seconds=5)
    assert result['status'] == 'unknown' and not result['checks'][-1]['passed']
    assert result['checks'][-1]['failure_reason'] == 'injected'


@pytest.mark.parametrize('options',[{'iterations':0},{'iterations':129},{'timeout_seconds':901},{'revision':'not-sha'}])
def test_qualification_bounds_reject_before_engine(options):
    with pytest.raises(ValueError):
        qualification.qualify_native(**options)


def test_fake_runtime_and_time_bound_prevent_qualification(monkeypatch):
    monkeypatch.setattr(qualification,'run_worker',lambda *a:{'annotations':{'provider':'fake'}})
    assert qualification.qualify_native(iterations=1)['status'] == 'unknown'
    def bound(*args):
        raise ResourceBound('injected_time_bound')
    monkeypatch.setattr(qualification,'run_worker',bound)
    result=qualification.qualify_native(iterations=1)
    assert result['status'] == 'unknown' and result['truncated']


def test_native_identity_rejects_standins_and_fingerprints_actual_files(tmp_path,monkeypatch):
    with pytest.raises(ValueError):
        runtime.native_identity(SimpleNamespace(__file__=str(tmp_path/'pyreason.py')))
    for name in ['__init__.py','pyreason.py','scripts/facts/fact.py','scripts/rules/rule.py',
                 'scripts/interpretation/interpretation.py','scripts/utils/rule_parser.py']:
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('SYNTHETIC UNREVIEWED IDENTITY FIXTURE')
    monkeypatch.setattr(runtime,'version',lambda name:'3.8.0' if name=='pyreason' else 'fixture')
    one=runtime.native_identity(SimpleNamespace(__file__=str(tmp_path/'__init__.py')))
    (tmp_path/'pyreason.py').write_text('changed synthetic fixture')
    two=runtime.native_identity(SimpleNamespace(__file__=str(tmp_path/'__init__.py')))
    assert one['engine_source_sha256'] != two['engine_source_sha256']
    assert one['identity_sha256'] != two['identity_sha256']
    import importlib.util
    monkeypatch.setattr(importlib.util,'find_spec',lambda _:None)
    with pytest.raises(ValueError):
        runtime.installed_identity()
    monkeypatch.setattr(importlib.util,'find_spec',lambda _:SimpleNamespace(origin=str(tmp_path/'__init__.py')))
    assert runtime.installed_identity() == two


def test_warmup_failures_block_ready_without_diagnostics(monkeypatch):
    monkeypatch.setattr(warmup,'run_worker',lambda *a:{'annotations':{'provider':'synthetic_test_only'}})
    monkeypatch.setattr(warmup,'infer_graph',lambda *a,**k:{'status':'computed','derived_facts':{'licensed':True}})
    assert warmup.warm_native()['status'] == 'ready'
    monkeypatch.setattr(warmup,'infer_graph',lambda *a,**k:{'status':'unknown','derived_facts':{}})
    result=warmup.warm_native()
    assert result['status']=='unknown' and not result['decision_authorized']
    with pytest.raises(ValueError):
        warmup.warm_native(timeout_seconds=0)


def test_recorded_native_corpus_has_fixed_schema_and_synthetic_labels():
    corpus=qualification.qualification_corpus()
    assert corpus['schema']=='vle-pyreason-native-corpus/v1' and len(corpus['activation_cases'])==162
    assert corpus['kind']=='synthetic' and corpus['review_status']=='unreviewed'
    for case in corpus['activation_cases']+corpus['graph_cases']:
        assert case['kind']=='synthetic' and case['review_status']=='unreviewed'
    # Shipped data must remain ordinary JSON, not executable fixtures.
    assert json.loads(json.dumps(corpus))==corpus
    assert (Path(__file__).resolve().parents[1]/'docs/PYREASON_NATIVE_PROTOCOL.md').is_file()


def test_bounded_command_line_reports_failures_without_success_exit(monkeypatch,capsys):
    import sys
    monkeypatch.setattr(sys,'argv',['warmup','--timeout-seconds','1'])
    monkeypatch.setattr(warmup,'warm_native',lambda **k:{'status':'unknown','decision_authorized':False})
    assert warmup.main()==1
    assert json.loads(capsys.readouterr().out)['status']=='unknown'
    monkeypatch.setattr(sys,'argv',['qualify','--iterations','1'])
    monkeypatch.setattr(qualification,'qualify_native',lambda **k:{'status':'unknown','decision_authorized':False})
    assert qualification.main()==1
    assert json.loads(capsys.readouterr().out)['decision_authorized'] is False
