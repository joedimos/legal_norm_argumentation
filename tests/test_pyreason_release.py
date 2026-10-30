"""Synthetic key material and invented attestations test gates, never qualify law."""
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.audit_bundle import canonical_json, digest
from src.pyreason_connector import digest as program_digest
from src.pyreason_qualification import qualification_corpus
from src.service import pyreason_release as policy
from src.service.app import create_app
from src.signed_annotation import sign_judgment
from tests.test_production_workflows import write_json
from tests.test_release_candidate import fixture_graph, rewrite_artifact
from tests.test_service import settings


def synthetic_backend_graph(tmp_path, monkeypatch):
    cfg, release = fixture_graph(tmp_path)
    cfg.pyreason_enabled = True
    cfg.pyreason_worker_mode = 'pooled'
    identity = {'identity_sha256':'c'*64, 'adapter_contract_sha256':'d'*64, 'provider':'native'}
    monkeypatch.setattr(policy, 'installed_identity', lambda: deepcopy(identity))
    native = {'schema':'vle-pyreason-qualification/v2','status':'passed','provider':'native', 'runtime':identity,
              'build_revision':cfg.build_revision, 'worker_mode':'pooled','corpus_sha256':program_digest(qualification_corpus()),
              'adapter_contract_sha256':identity['adapter_contract_sha256'], 'native_activation_cases':162,'native_graph_cases':7,
              'checks':[{'name':name,'passed':True} for name in sorted(
                  {f'activation_{kind}_{i}' for kind in ('batch','values') for i in range(6)}
                  | {c['id'] for c in qualification_corpus()['graph_cases']}
                  | {'derived_fact_controls_compliance'} | {f'performance_result_{i}' for i in range(32)})], 'bounds':{'iterations':32},
              'performance':{'native':{'count':32,'samples_ms':[1]*32},'norm_counts':[1,8,32,64]*8}}
    native_sha = write_json(tmp_path/'native.json', native)
    benchmark = json.loads((tmp_path/'benchmark.json').read_text())
    assessment = json.loads((tmp_path/'annotations.json').read_text())
    target = {'schema':'vle-pyreason-qualification-target/v1','build_revision':cfg.build_revision,'worker_mode':'pooled',
              'runtime_sha256':identity['identity_sha256'],'native_sha256':native_sha,
              'predictions_sha256':benchmark['predictions_sha256'],'packet_sha256':assessment['packet']['packet_sha256']}
    target_sha = digest(canonical_json(target))
    for i, submission in enumerate(assessment['submissions']):
        payload = {**submission['payload'], 'target_sha256':target_sha}
        assessment['submissions'][i] = sign_judgment(payload,packet_sha256=assessment['packet']['packet_sha256'],
                        role='annotator',private_key_file=tmp_path/f'person-{i}.reviewer.key')
    rewrite_artifact(tmp_path,cfg,release,'annotations',lambda x: x.update(assessment))
    rewrite_artifact(tmp_path,cfg,release,'benchmark',lambda x: x.update(reasoning_backend='pyreason',backend_target_sha256=target_sha,
                                  independent_assessment_sha256=digest(canonical_json(assessment))))
    rewrite_artifact(tmp_path,cfg,release,'load',lambda x: x.update(reasoning_backend='pyreason',worker_mode='pooled',backend_computations_checked=True))
    manifest = {'schema':'vle-pyreason-release/v1','scope':'fr-us-ip-supervised/v1','build_revision':cfg.build_revision,
                'deployment_id':release['deployment_id'],'expires_at':(datetime.now(UTC)+timedelta(days=1)).isoformat(),
                'worker_mode':'pooled','approved_by':['SYNTHETIC-custodian-a','SYNTHETIC-custodian-b'],
                'release_sha256':cfg.release_evidence_sha256,'native':{'filename':'native.json','sha256':native_sha}}
    cfg.pyreason_release_path = str(tmp_path/'native-release.json')
    cfg.pyreason_release_sha256 = write_json(Path(cfg.pyreason_release_path),manifest)
    return cfg,manifest,release


def test_backend_gate_requires_all_independent_and_native_artifacts(tmp_path,monkeypatch):
    cfg,m,release = synthetic_backend_graph(tmp_path,monkeypatch)
    from jsonschema import validate
    root=Path(__file__).resolve().parents[1]
    validate(m,json.loads((root/'schemas/pyreason-release.schema.json').read_text()))
    result = policy.pyreason_release_gate(cfg)
    assert result['passed'] and result['release_qualified'] and not result['legal_certification'], result
    cfg.pyreason_worker_mode = 'isolated'
    assert not policy.pyreason_release_gate(cfg)['passed']
    cfg.pyreason_worker_mode = 'pooled'
    registry = json.loads((tmp_path/'registry.json').read_text())
    registry['reviewers'][0]['revoked'] = True
    cfg.reviewer_registry_sha256 = write_json(tmp_path/'registry.json',registry)
    assert not policy.pyreason_release_gate(cfg)['passed']


@pytest.mark.parametrize('field,value',[
    ('build_revision','0'*40), ('release_sha256','0'*64), ('expires_at','2020-01-01T00:00:00+00:00'),
    ('approved_by',['single']), ('native',{'filename':'../escape.json','sha256':'a'*64}), ('deployment_id','different'),
])
def test_backend_manifest_staleness_path_or_scope_mismatch_blocks(tmp_path,monkeypatch,field,value):
    cfg,m,_ = synthetic_backend_graph(tmp_path,monkeypatch)
    m[field] = value
    cfg.pyreason_release_sha256 = write_json(Path(cfg.pyreason_release_path),m)
    assert not policy.pyreason_release_gate(cfg)['passed']


@pytest.mark.parametrize('field,value', [('provider','fake'),('status','unknown'),('native_activation_cases',161),('native_graph_cases',0),
                                      ('truncated',True),('checks',[{'passed':False}]),('corpus_sha256','0'*64)])
def test_unknown_fake_or_incomplete_native_evidence_is_rejected(tmp_path,monkeypatch,field,value):
    cfg,m,_ = synthetic_backend_graph(tmp_path,monkeypatch)
    native = json.loads((tmp_path/'native.json').read_text())
    native[field] = value
    m['native']['sha256'] = write_json(tmp_path/'native.json',native)
    cfg.pyreason_release_sha256 = write_json(Path(cfg.pyreason_release_path),m)
    assert not policy.pyreason_release_gate(cfg)['passed']


def test_legacy_review_override_cannot_qualify_native_and_paths_require_pins(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        settings(tmp_path,pyreason_release_path='some-file.json')
    cfg = settings(tmp_path,pyreason_enabled=True,release_enforcement='reviewed')
    assert not policy.pyreason_release_gate(cfg)['passed']
    assert settings(tmp_path,pyreason_release_path='',pyreason_release_sha256='').pyreason_release_path is None


def test_readiness_waits_for_native_startup_and_does_not_authorize(tmp_path,monkeypatch):
    from src import pyreason_warmup
    monkeypatch.setattr(pyreason_warmup,'warm_native',lambda **kwargs:{'status':'unknown'})
    cfg = settings(tmp_path,pyreason_enabled=True,pyreason_warmup=True,pyreason_worker_mode='pooled')
    with TestClient(create_app(cfg)) as client:
        assert client.get('/health/live').status_code == 200
        assert client.get('/health/ready').status_code == 503
    monkeypatch.setattr(pyreason_warmup,'warm_native',lambda **kwargs:{'status':'ready'})
    with TestClient(create_app(cfg)) as client:
        assert client.get('/health/ready').status_code == 200


def test_native_probe_never_counts_session_only_or_unknown_as_evaluation_load():
    from src.service.operations import load_probe
    class Probe:
        def __init__(self):
            self.status = 'computed'
        def call(self,path,**kwargs):
            assert kwargs['payload']['reasoning_backend'] == 'pyreason'
            return {'decision_authorized':False,'compliance':{'status':'compliant','reasoning':{'backend':{'status':self.status,'worker_mode':'pooled'}}}}
    probe = Probe()
    result = load_probe(probe,'synthetic',count=2,concurrency=1,evaluation_source='id',reasoning_backend='pyreason',worker_mode='pooled')
    assert result['passed'] and result['backend_computations_checked']
    probe.status = 'unknown'
    assert not load_probe(probe,'synthetic',count=1,evaluation_source='id',reasoning_backend='pyreason',worker_mode='pooled')['passed']


def test_full_service_native_authorization_requires_both_backend_and_exact_case_gates(tmp_path,monkeypatch):
    # Test keys, fabricated reports and mocked engine are explicitly synthetic.
    from tests.test_production_workflows import post, signed_packet, source_review
    from tests.test_pyreason_normative import worker
    from tests.test_service import REVIEWER_TOKEN, ingest
    # The existing worker fixture function supplies a deterministic recorded engine.
    worker.__wrapped__(monkeypatch)
    cfg,m,release=synthetic_backend_graph(tmp_path,monkeypatch)
    monkeypatch.setattr('src.service.app.source_scope_gate',lambda *a:{'passed':True,'basis':'synthetic_gate_mechanics_only'})
    with TestClient(create_app(cfg)) as client:
        source=ingest(client,'The operator must retain logs.')
        source_review(client,source)
        body={'reasoning_backend':'pyreason','high_stakes':True,'actor':'operator','at_date':'2026-10-27',
              'capacity':'research','jurisdiction':'FR','observed_actions':['retain logs']}
        target=post(client,f"/v1/sources/{source['id']}/release-target",body).json()['target_sha256']
        before=post(client,f"/v1/sources/{source['id']}/evaluate",body)
        assert before.status_code==409 and not before.json()['decision_authorized']
        signed=signed_packet(tmp_path,'The operator must retain logs.',target=target)
        uploaded=post(client,f"/v1/sources/{source['id']}/legal-assessments",{'target_sha256':target,'assessment':signed},REVIEWER_TOKEN)
        assert uploaded.status_code==201, uploaded.text
        body['legal_assessment_id']=uploaded.json()['id']
        response=post(client,f"/v1/sources/{source['id']}/evaluate",body,key='synthetic-native-qualification')
        assert response.status_code==200, response.text
        assert response.json()['decision_authorized'] and response.json()['production_release']['reasoning_backend']['passed']
        cfg.pyreason_release_sha256='0'*64
        revoked=post(client,f"/v1/sources/{source['id']}/evaluate",body,key='synthetic-native-qualification')
        assert revoked.status_code==409 and not revoked.json()['decision_authorized']
