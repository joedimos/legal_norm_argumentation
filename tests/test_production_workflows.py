from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.audit_bundle import canonical_json, digest, verify_bundle
from src.legal_annotation import DIMENSIONS, prepare_annotation
from src.live_generation import generate_candidate
from src.service.app import create_app
from src.service.db_models import WorkflowRecord
from src.service.operations import ProbeClient, load_probe
from src.service.release_policy import deployment_release_gate
from src.service.workflows import add_event, create_workflow, process_one
from src.signed_annotation import create_reviewer_key, sign_judgment
from tests.test_live_generation import proposal, transport
from tests.test_response_alignment import candidate, family, ref, snapshot
from tests.test_service import OPERATOR_TOKEN, REVIEWER_TOKEN, auth, ingest, settings

BUILD = 'b' * 40


def post(client, url, body, token=OPERATOR_TOKEN, key=None):
    return client.post(url, json=body, headers={**auth(token), 'Idempotency-Key': key or str(uuid.uuid4())})


def source_review(client, source):
    for item in source['candidates']:
        assert post(client, f"/v1/sources/{source['id']}/reviews", {'candidate_id': item['id'], 'decision': 'approved'},
                    REVIEWER_TOKEN).status_code == 201


def base_family(client):
    value = family()
    value.pop('reviews')
    response = post(client, '/v1/workflows/families', {'family': value})
    assert response.status_code == 201, response.text
    id = response.json()['id']
    assert post(client, f'/v1/workflows/{id}/reviews', {'scope': 'family', 'decision': 'approved', 'notes': 'Fixture source review only.'},
                REVIEWER_TOKEN).status_code == 201
    return id


def test_authenticated_workflow_generation_review_persistence_and_dedup(tmp_path):
    cfg = settings(tmp_path, inference_enabled=True, model_name='test-model')
    app = create_app(cfg)
    with TestClient(app) as client:
        id = base_family(client)
        body = {'family_workflow_id': id, 'prompt': 'Handle records.'}
        job = post(client, '/v1/workflows/generations', body)
        assert job.status_code == 202
        job_id = job.json()['id']
        assert post(client, '/v1/workflows/generations', body).json()['id'] == job_id
        calls = []
        assert process_one(app.state.db, cfg, transport=transport(proposal(), calls))['status'] == 'review_required'
        assert process_one(app.state.db, cfg, transport=transport(proposal(), calls))['status'] == 'idle'
        assert len(calls) == 1
        bad_review = post(client, f'/v1/workflows/{job_id}/reviews', {'scope': 'behavior', 'decision': 'approved', 'notes': 'Self approval'}, OPERATOR_TOKEN)
        assert bad_review.status_code == 403
        review = post(client, f'/v1/workflows/{job_id}/reviews', {'scope': 'behavior', 'decision': 'approved', 'notes': 'Fixture candidate mapping review.'}, REVIEWER_TOKEN)
        assert review.status_code == 201
        assert review.json()['review_request']['reviews']['behavior']['reviewer'] == 'reviewer-b'
        archive = client.get(f'/v1/workflows/{job_id}/audit-bundle', headers=auth(REVIEWER_TOKEN))
        assert verify_bundle(archive.content, expected_manifest_digest=archive.headers['X-Audit-Manifest-SHA256'])['valid']
    # A newly constructed service reads persisted job/review state.
    with TestClient(create_app(cfg)) as client:
        assert client.get(f'/v1/workflows/{job_id}', headers=auth(OPERATOR_TOKEN)).json()['review_request']['reviews']['behavior']['decision'] == 'approved'


def test_claimed_job_never_retried_after_worker_crash(tmp_path):
    cfg = settings(tmp_path, inference_enabled=True, model_name='test-model')
    app = create_app(cfg)
    with TestClient(app) as client:
        family_id = base_family(client)
        job_id = post(client, '/v1/workflows/generations', {'family_workflow_id': family_id, 'prompt': 'Handle records.'}).json()['id']
        with app.state.db.session() as session:
            job = session.get(WorkflowRecord, job_id)
            add_event(session, job, 'claim', 'crashed-worker', {'status': 'claimed'}, 'fixture')
        calls = []
        assert process_one(app.state.db, cfg, transport=transport(proposal(), calls))['status'] == 'idle'
        assert calls == []
        assert client.get(f'/v1/workflows/{job_id}', headers=auth(OPERATOR_TOKEN)).json()['status'] == 'running_or_interrupted'


def test_caller_cannot_override_runtime_or_forge_family_approval(tmp_path):
    with TestClient(create_app(settings(tmp_path, inference_enabled=True, model_name='test-model'))) as client:
        assert post(client, '/v1/workflows/families', {'family': family()}).status_code == 409
        id = base_family(client)
        assert post(client, '/v1/workflows/generations', {'family_workflow_id': id, 'prompt': 'x', 'endpoint': 'https://attacker.invalid'}).status_code == 422
        assert post(client, '/v1/workflows/families', {'family': {'schema': 'invalid'}}).status_code == 409
        assert client.get('/workflows').status_code == 200
        assert 'textContent' in client.get('/workflow-console.js').text


def test_current_family_revocation_blocks_queued_model_call(tmp_path):
    cfg = settings(tmp_path, inference_enabled=True, model_name='test-model')
    app = create_app(cfg)
    with TestClient(app) as client:
        id = base_family(client)
        post(client, '/v1/workflows/generations', {'family_workflow_id': id, 'prompt': 'Handle records.'})
        assert post(client, f'/v1/workflows/{id}/reviews', {'scope': 'family', 'decision': 'rejected', 'notes': 'New source finding.'}, REVIEWER_TOKEN).status_code == 201
        calls = []
        assert process_one(app.state.db, cfg, transport=transport(proposal(), calls))['status'] == 'worker_failed'
        assert calls == []


def enrollment(tmp_path):
    now = datetime.now(UTC)
    registry = {'schema': 'vle-reviewer-registry/v1', 'excluded_independence_groups': ['test-generator'], 'reviewers': []}
    for i in range(2):
        public = create_reviewer_key(tmp_path / f'person-{i}.reviewer.key', tmp_path / f'person-{i}.pub')['public_key']
        registry['reviewers'].append({'id': f'external-fixture-{i}', 'public_key': public, 'roles': ['annotator'], 'jurisdictions': ['FR'],
             'independence_group': f'fixture-group-{i}', 'valid_from': '2026-01-01', 'valid_until': '2030-01-01', 'revoked': False,
             'credential_check': {'checked_by': 'test-custodian', 'checked_at': (now - timedelta(days=1)).isoformat(), 'evidence_sha256': 'a' * 64}})
    return registry


def signed_packet(tmp_path, source_text, *, target=None):
    base = family()
    base.pop('reviews')
    base['context'].update(actor='operator')
    document = snapshot('source', 'normative', source_text)
    base['documents'] = [document]
    base['interpretations'][0]['source_refs'] = [ref(document)]
    draft = candidate(base, 'answer', False, reviewed=False)
    from src.response_alignment import bind_candidate
    observation = snapshot('fixture-observation', 'observation', 'Synthetic fixture: the operator retained logs.')
    base['documents'].append(observation)
    bound = bind_candidate(base, draft)
    packet = prepare_annotation({'schema': 'vle-annotation-study/v1', 'study_id': 'SYNTHETIC-TEST-ONLY', 'units': [
        {'id': 'one', 'source_group': 'fixture-provision', 'split': 'test', 'prompt': 'Synthetic fixture.', 'family': bound}]})
    labels = dict.fromkeys(DIMENSIONS, 'supported')
    labels.update(grounding='supported', harmful_assistance='unsupported')
    submissions = []
    for i in range(2):
        payload = {'reviewer': {'id': f'external-fixture-{i}', 'qualification': 'Synthetic test identity; not an expert.',
                   'independent': True, 'conflicts_disclosed': False}, 'packet_sha256': packet['packet_sha256'], 'annotations': [
                     {'unit_id': 'one', 'unit_sha256': packet['units'][0]['unit_sha256'], 'judgment': {
                       'labels': labels, 'rationale': 'Synthetic regression evidence only.', 'alternative_interpretations': [],
                       'evidence_refs': [ref(document), ref(draft['document']), ref(observation)]}}]}
        if target:
            payload['target_sha256'] = target
        submissions.append(sign_judgment(payload, packet_sha256=packet['packet_sha256'], role='annotator', private_key_file=tmp_path / f'person-{i}.reviewer.key'))
    return {'schema': 'vle-signed-assessment/v1', 'packet': packet, 'submissions': submissions}


def write_json(path, value):
    path.write_bytes(canonical_json(value))
    return digest(canonical_json(value))


def release_manifest(tmp_path, cfg, signed):
    benchmark = {'schema': 'vle-production-benchmark/v1', 'build_revision': BUILD, 'split': 'test', 'case_count': 1,
                 'extraction_f1': 1.0, 'compliance_accuracy': 1.0, 'corpus_sha256': signed['packet']['packet_sha256'],
                 'independent_assessment_sha256': digest(canonical_json(signed))}
    model = generate_candidate({'schema': 'vle-live/v1', 'family': family(), 'prompt': 'Synthetic test only.', 'model': 'test-model'},
                               transport=transport(proposal(), []))
    reports = {'benchmark': benchmark, 'annotations': signed, 'model': model}
    for kind in ('smoke', 'load', 'restore', 'legifrance'):
        reports[kind] = {'schema': 'vle-operational-evidence/v1', 'kind': kind, 'passed': True, 'deployment_id': 'test-only', 'build_revision': BUILD}
    reports['smoke']['audit_verified'] = True
    reports['restore']['snapshot_sha256'] = 'a' * 64
    reports['legifrance']['corpora'] = {'ip': 1, 'competition': 1}
    reports['load'].update(coverage='authenticated_evaluation_with_audit_writes', requests=100, errors=0, p95_ms=1.0)
    manifest = {'schema': 'vle-release-evidence/v1' , 'build_revision': BUILD, 'deployment_id': 'test-only',
                'expires_at': (datetime.now(UTC) + timedelta(days=1)).isoformat(), 'approved_by': ['fixture-custodian-a', 'fixture-custodian-b'],
                'evidence': {kind: {'filename': kind + '.json', 'sha256': write_json(tmp_path / (kind + '.json'), value)} for kind, value in reports.items()}}
    cfg.release_evidence_path = str(tmp_path / 'release.json')
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)
    return manifest


def test_release_and_case_gates_positive_staleness_and_idempotent_revocation(tmp_path):
    registry = enrollment(tmp_path)
    registry_path = tmp_path / 'registry.json'
    cfg = settings(tmp_path, build_revision=BUILD, model_name='test-model', release_min_cases=1,
                   reviewer_registry_path=str(registry_path), reviewer_registry_sha256=write_json(registry_path, registry))
    app = create_app(cfg)
    with TestClient(app) as client:
        source = ingest(client, 'The operator must retain logs.')
        source_review(client, source)
        context = {'observed_actions': ['retain logs'], 'high_stakes': True, 'actor': 'operator',
                   'capacity': 'research', 'jurisdiction': 'FR', 'at_date': '2026-10-27'}
        target = post(client, f"/v1/sources/{source['id']}/release-target", context).json()['target_sha256']
        signed = signed_packet(tmp_path, 'The operator must retain logs.', target=target)
        release_manifest(tmp_path, cfg, signed)
        assert deployment_release_gate(cfg)['passed']
        uploaded = post(client, f"/v1/sources/{source['id']}/legal-assessments", {'target_sha256': target, 'assessment': signed}, REVIEWER_TOKEN)
        assert uploaded.status_code == 201, uploaded.text
        context['legal_assessment_id'] = uploaded.json()['id']
        evaluation = post(client, f"/v1/sources/{source['id']}/evaluate", context, key='positive-release-test')
        assert evaluation.status_code == 200, evaluation.text
        assert evaluation.json()['decision_authorized'] is True
        changed = post(client, f"/v1/sources/{source['id']}/evaluate", {**context, 'observed_actions': []})
        assert changed.status_code == 409
        # A current trusted registry can revoke a previously accepted signer.
        registry['reviewers'][0]['revoked'] = True
        cfg.reviewer_registry_sha256 = write_json(registry_path, registry)
        revoked = post(client, f"/v1/sources/{source['id']}/evaluate", context, key='positive-release-test')
        assert revoked.status_code == 409 and revoked.json()['decision_authorized'] is False


def test_research_only_flag_never_becomes_production_authorization(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        source = ingest(client, 'The operator must retain logs.')
        source_review(client, source)
        evaluated = post(client, f"/v1/sources/{source['id']}/evaluate", {'high_stakes': False, 'observed_actions': ['retain logs']})
        assert evaluated.status_code == 200
        assert evaluated.json()['authorization_scope'] == 'research_only'
        assert evaluated.json()['decision_authorized'] is False


def test_release_artifacts_missing_expired_or_threshold_failure_block(tmp_path):
    registry = enrollment(tmp_path)
    registry_path = tmp_path / 'registry.json'
    cfg = settings(tmp_path, build_revision=BUILD, model_name='test-model', release_min_cases=1,
                   reviewer_registry_path=str(registry_path), reviewer_registry_sha256=write_json(registry_path, registry))
    signed = signed_packet(tmp_path, 'The operator must retain logs.')
    manifest = release_manifest(tmp_path, cfg, signed)
    assert deployment_release_gate(cfg)['passed']
    cfg.release_min_cases = 100
    assert not deployment_release_gate(cfg)['passed']
    cfg.release_min_cases = 1
    manifest['expires_at'] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)
    assert not deployment_release_gate(cfg)['passed']


def test_migration_workflow_immutability_postgres_when_configured(tmp_path):
    url = os.environ.get('VLE_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('PostgreSQL CI only')
    cfg = settings(tmp_path, database_url=url, auto_create_schema=False)
    app = create_app(cfg)
    with app.state.db.session() as session:
        workflow = create_workflow(session, 'family', 'fixture-postgres', {'test': str(uuid.uuid4())}, 'pg-fixture')
        add_event(session, workflow, 'test', 'fixture-postgres', {'ok': True}, 'pg-fixture')
        id = workflow.id
    assert app.state.db.ready(require_migrations=True)
    with pytest.raises(DBAPIError, match="append-only"), app.state.db.session() as session:
        session.execute(text('DELETE FROM workflow_requests WHERE id=:id'), {'id': id})
    with pytest.raises(DBAPIError, match="append-only"), app.state.db.session() as session:
        session.execute(text('UPDATE workflow_events SET actor=:actor WHERE workflow_id=:id'), {'actor': 'changed', 'id': id})


def test_load_probe_and_origin_restrictions():
    class Client:
        def call(self, *args, **kwargs):
            return {'ok': True}
    assert load_probe(Client(), 'fixture-token', count=10, concurrency=2)['passed']
    with pytest.raises(ValueError):
        ProbeClient('http://untrusted.example')
    with pytest.raises(ValueError):
        load_probe(Client(), 'fixture-token', count=1001)


def test_evaluation_target_changes_with_facts_and_new_review(tmp_path):
    with TestClient(create_app(settings(tmp_path, build_revision=BUILD))) as client:
        source = ingest(client, 'The operator must retain logs.')
        source_review(client, source)
        url = f"/v1/sources/{source['id']}/release-target"
        base = post(client, url, {'observed_actions': ['retain logs']}).json()['target_sha256']
        assert post(client, url, {'observed_actions': ['retain logs'], 'facts': {'condition': True}}).json()['target_sha256'] != base
        source_review(client, source)
        assert post(client, url, {'observed_actions': ['retain logs']}).json()['target_sha256'] != base


def test_release_artifact_tampering_and_weak_load_evidence_rejected(tmp_path):
    registry = enrollment(tmp_path)
    registry_path = tmp_path / 'registry.json'
    cfg = settings(tmp_path, build_revision=BUILD, model_name='test-model', release_min_cases=1,
                   reviewer_registry_path=str(registry_path), reviewer_registry_sha256=write_json(registry_path, registry))
    manifest = release_manifest(tmp_path, cfg, signed_packet(tmp_path, 'The operator must retain logs.'))
    model = json.loads((tmp_path / 'model.json').read_text())
    model['model'] = 'unvalidated-model'
    write_json(tmp_path / 'model.json', model)
    assert not deployment_release_gate(cfg)['passed']
    # Even repinning the modified graph cannot substitute a different model.
    manifest['evidence']['model']['sha256'] = write_json(tmp_path / 'model.json', model)
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)
    assert not deployment_release_gate(cfg)['passed']
    manifest = release_manifest(tmp_path, cfg, signed_packet(tmp_path, 'The operator must retain logs.'))
    load = json.loads((tmp_path / 'load.json').read_text())
    load['coverage'] = 'authenticated_session_endpoint_only'
    manifest['evidence']['load']['sha256'] = write_json(tmp_path / 'load.json', load)
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)
    assert not deployment_release_gate(cfg)['passed']


def test_signed_upload_cannot_bind_unsigned_target_or_wrong_source(tmp_path):
    registry = enrollment(tmp_path)
    registry_path = tmp_path / 'registry.json'
    cfg = settings(tmp_path, reviewer_registry_path=str(registry_path), reviewer_registry_sha256=write_json(registry_path, registry))
    with TestClient(create_app(cfg)) as client:
        source = ingest(client, 'The operator must retain logs.')
        body = {'target_sha256': 'a' * 64, 'assessment': signed_packet(tmp_path, 'The operator must retain logs.')}
        assert post(client, f"/v1/sources/{source['id']}/legal-assessments", body, REVIEWER_TOKEN).status_code == 409
        body['assessment'] = signed_packet(tmp_path, 'A different source must be checked.', target='a' * 64)
        assert post(client, f"/v1/sources/{source['id']}/legal-assessments", body, REVIEWER_TOKEN).status_code == 409


def test_failed_model_call_is_terminal_and_does_not_repeat(tmp_path):
    cfg = settings(tmp_path, inference_enabled=True, model_name='test-model')
    app = create_app(cfg)
    with TestClient(app) as client:
        id = base_family(client)
        post(client, '/v1/workflows/generations', {'family_workflow_id': id, 'prompt': 'Handle records.'})
        calls = []
        def broken(*args, **kwargs):
            calls.append(1)
            raise ValueError('do not expose provider secrets')
        assert process_one(app.state.db, cfg, transport=broken)['status'] == 'generation_failed'
        assert process_one(app.state.db, cfg, transport=broken)['status'] == 'idle'
        assert len(calls) == 1


def test_unknown_schema_revision_is_not_ready(tmp_path):
    from src.service.database import Database
    database = Database(settings(tmp_path))
    database.create_schema_for_tests()
    with database.engine.begin() as connection:
        connection.execute(text('CREATE TABLE alembic_version (version_num VARCHAR(64))'))
        connection.execute(text("INSERT INTO alembic_version VALUES ('0001_production_service')"))
    assert not database.ready(require_migrations=True)
    database.dispose()


def test_complete_service_to_worker_to_http_model_flow(tmp_path):
    from tests.test_model_runtime import runtime
    with runtime() as (base, http_events):
        cfg = settings(tmp_path, inference_enabled=True, model_provider='vllm', model_name='test-model',
                       model_endpoint=base + '/v1/chat/completions')
        app = create_app(cfg)
        with TestClient(app) as client:
            id = base_family(client)
            job = post(client, '/v1/workflows/generations', {'family_workflow_id': id, 'prompt': 'Handle records.'}).json()['id']
            assert process_one(app.state.db, cfg)['status'] == 'review_required'
            result = client.get(f'/v1/workflows/{job}', headers=auth(OPERATOR_TOKEN)).json()
            assert result['result']['candidate']['document']['text'] == proposal()['text']
            assert http_events[0][0] == 'POST'
            assert client.get('/v1/release-status', headers=auth(OPERATOR_TOKEN)).json()['release']['passed'] is False


def test_parallel_worker_claim_postgres_when_configured(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    url = os.environ.get('VLE_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('PostgreSQL CI only')
    cfg = settings(tmp_path, database_url=url, auto_create_schema=False, inference_enabled=True, model_name='test-model')
    app = create_app(cfg)
    with TestClient(app) as client:
        id = base_family(client)
        post(client, '/v1/workflows/generations', {'family_workflow_id': id, 'prompt': 'Parallel claim fixture.'})
        calls = []
        def invoke(_):
            return process_one(app.state.db, cfg, transport=transport(proposal(), calls))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(invoke, range(2)))
        assert len(calls) == 1
        assert sum(r['status'] == 'review_required' for r in results) == 1


def test_operational_smoke_exercises_integrated_api_and_audit(tmp_path):
    from src.service.app_with_ui import create_app as integrated_app
    from src.service.operations import smoke
    with TestClient(integrated_app(settings(tmp_path))) as client:
        class Adapter:
            def call(self, path, *, token=None, payload=None, binary=False):
                response = post(client, path, payload, token) if payload is not None else client.get(path, headers=auth(token) if token else {})
                assert response.status_code in {200, 201}, response.text
                return (response.content, response.headers) if binary else response.json()
        result = smoke(Adapter(), OPERATOR_TOKEN, REVIEWER_TOKEN, REVIEWER_TOKEN)
        assert result['audit_verified']


def test_sqlite_migration_and_snapshot_restore_comparison(tmp_path):
    import shutil
    import subprocess
    import sys

    from src.service.database import Database
    from src.service.operations import database_snapshot
    url = f'sqlite+pysqlite:///{tmp_path / "before.db"}'
    env = {**os.environ, 'VLE_DATABASE_URL': url}
    run = subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', 'head'], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    original = Database(settings(tmp_path, database_url=url, auto_create_schema=False))
    with original.session() as session:
        create_workflow(session, 'family', 'migration-test', {'fixture': 'restore-roundtrip'}, 'restore-fixture')
    baseline = database_snapshot(original)
    original.dispose()
    shutil.copyfile(tmp_path / 'before.db', tmp_path / 'after.db')
    restored = Database(settings(tmp_path, database_url=f'sqlite+pysqlite:///{tmp_path / "after.db"}', auto_create_schema=False))
    assert database_snapshot(restored) == baseline
    with restored.session() as session:
        create_workflow(session, 'family', 'migration-test', {'fixture': 'added-after-backup'}, 'restore-fixture')
    assert database_snapshot(restored) != baseline
    restored.dispose()


def test_explicit_failed_job_retry_is_audited_and_deduplicated(tmp_path):
    cfg = settings(tmp_path, inference_enabled=True, model_name='test-model')
    app = create_app(cfg)
    with TestClient(app) as client:
        id = base_family(client)
        body = {'family_workflow_id': id, 'prompt': 'Explicit retry fixture.'}
        first = post(client, '/v1/workflows/generations', body).json()['id']
        def broken(*args, **kwargs):
            raise ValueError('fixture transport failure')
        process_one(app.state.db, cfg, transport=broken)
        assert post(client, '/v1/workflows/generations', body).json()['id'] == first
        retry = {**body, 'retry_of': first, 'retry_reason': 'Provider inspected; prior execution accounted for.'}
        second = post(client, '/v1/workflows/generations', retry)
        assert second.status_code == 202
        assert second.json()['id'] != first
        assert post(client, '/v1/workflows/generations', retry).json()['id'] == second.json()['id']
        calls = []
        assert process_one(app.state.db, cfg, transport=transport(proposal(), calls))['status'] == 'review_required'
        assert len(calls) == 1


def test_default_reviewed_mode_removes_external_validation_block_but_keeps_source_review(tmp_path):
    from src.service.config import Settings
    payload = settings(tmp_path).model_dump()
    payload.pop('release_enforcement')
    cfg = Settings(**payload, _env_file=None)
    with TestClient(create_app(cfg)) as client:
        source = ingest(client, 'The operator must retain logs.')
        body = {'high_stakes': True, 'observed_actions': ['retain logs']}
        url = f"/v1/sources/{source['id']}/evaluate"
        assert post(client, url, body).status_code == 409
        source_review(client, source)
        response = post(client, url, body)
        assert response.status_code == 200, response.text
        assert response.json()['decision_authorized'] is True
        assert response.json()['authorization_scope'] == 'reviewed_operation'
        release = response.json()['production_release']
        assert release['enforcement'] == 'reviewed' and release['override_used']
        assert not release['external_validation_satisfied']
        assert not release['deployment']['passed'] and not release['independent_case_review']['passed']
        assert not release['legal_certification']
        status = client.get('/v1/release-status', headers=auth(OPERATOR_TOKEN)).json()
        assert status['release_enforcement'] == 'reviewed' and not status['case_review_required']
        archive = client.get(f"/v1/runs/{response.json()['run_id']}/audit-bundle", headers=auth(REVIEWER_TOKEN))
        assert verify_bundle(archive.content)['valid']


def test_reviewed_mode_never_promotes_research_only_evaluation(tmp_path):
    with TestClient(create_app(settings(tmp_path, release_enforcement='reviewed'))) as client:
        source = ingest(client, 'The operator must retain logs.')
        source_review(client, source)
        result = post(client, f"/v1/sources/{source['id']}/evaluate", {'high_stakes': False, 'observed_actions': ['retain logs']})
        assert result.status_code == 200 and not result.json()['decision_authorized']
        assert not result.json()['production_release']['override_used']


def test_policy_changes_do_not_replay_historical_approval_as_new_authorization(tmp_path):
    cfg = settings(tmp_path, release_enforcement='reviewed')
    with TestClient(create_app(cfg)) as client:
        source = ingest(client, 'The operator must retain logs.')
        source_review(client, source)
        body = {'high_stakes': True, 'observed_actions': ['retain logs']}
        url = f"/v1/sources/{source['id']}/evaluate"
        assert post(client, url, body, key='policy-replay-fixture').status_code == 200
        cfg.release_enforcement = 'strict'
        blocked = post(client, url, body, key='policy-replay-fixture')
        assert blocked.status_code == 409 and not blocked.json()['decision_authorized']
        cfg.release_enforcement = 'reviewed'
        source_review(client, source)
        # A changed review keeps the same reviewed-operation policy. A negative
        # review must still block, including a previously authorized cache entry.
        post(client, f"/v1/sources/{source['id']}/reviews", {'candidate_id': source['candidates'][0]['id'], 'decision': 'rejected'}, REVIEWER_TOKEN)
        assert post(client, url, body, key='policy-replay-fixture').status_code == 409
