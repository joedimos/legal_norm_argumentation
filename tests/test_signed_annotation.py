from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime

import pytest

pytest.importorskip('cryptography', reason='optional annotation extra is tested by research-verification CI')

from src.audit_bundle import canonical_json, digest, verify_bundle
from src.legal_annotation import annotation_template, assess_annotations
from src.signed_annotation import assess_signed_annotations, create_reviewer_key, sign_judgment
from tests.test_legal_annotation import assessment, reviewer

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)


@pytest.fixture
def signed(tmp_path):
    plain = assessment()
    registry = {'schema': 'vle-reviewer-registry/v1', 'excluded_independence_groups': ['generator-team'], 'reviewers': []}
    envelopes = []
    for i, submission in enumerate(plain['submissions']):
        # Synthetic test identities only. No fabricated expert dataset is shipped.
        key = tmp_path / f'test-{i}.key'
        public = create_reviewer_key(key, tmp_path / f'test-{i}.pub')['public_key']
        submission['annotations'][0]['judgment']['evidence_refs'] = []
        registry['reviewers'].append({
            'id': submission['reviewer']['id'], 'public_key': public, 'roles': ['annotator'], 'jurisdictions': ['FR'],
            'independence_group': f'test-group-{i}', 'valid_from': '2026-01-01', 'valid_until': '2027-01-01',
            'revoked': False, 'credential_check': {'checked_by': 'fixture-only', 'checked_at': '2026-01-01T00:00:00+00:00',
                                                  'evidence_sha256': 'a' * 64}})
        envelopes.append(sign_judgment(submission, packet_sha256=plain['packet']['packet_sha256'], role='annotator',
                                       private_key_file=key, now=NOW))
    request = {'schema': 'vle-signed-assessment/v1', 'packet': plain['packet'], 'submissions': envelopes}
    return request, registry


def check(request, registry):
    return assess_signed_annotations(request, registry=registry, expected_registry_sha256=digest(canonical_json(registry)),
                                     expected_packet_sha256=request['packet']['packet_sha256'], now=NOW)


def test_signed_independent_records_roundtrip_without_creating_legal_approvals(signed):
    request, registry = signed
    result = check(request, registry)
    assert result['status'] == 'annotations_collected'
    assert result['key_signatures_verified']
    assert not result['qualifications_verified'] and not result['legal_certification']
    assert result['signed_evidence'] == request
    assert 'reviews' not in result


def test_key_permissions_no_overwrite_and_incomplete_templates(tmp_path, signed):
    key, pub = tmp_path / 'person.key', tmp_path / 'person.pub'
    create_reviewer_key(key, pub)
    assert os.stat(key).st_mode & 0o777 == 0o600
    original = key.read_bytes()
    pub.unlink()
    with pytest.raises(FileExistsError):
        create_reviewer_key(key, pub)
    assert key.read_bytes() == original and not pub.exists()
    request, _ = signed
    template = annotation_template(request['packet'], reviewer_id='new reviewer', qualification='Unverified claim')
    assert template['annotations'][0]['judgment']['labels']['faithfulness'] is None
    with pytest.raises(ValueError):
        assess_annotations({'schema': 'vle-annotation-assessment/v1', 'packet': request['packet'],
                            'submissions': [template, request['submissions'][0]['payload']]})


@pytest.mark.parametrize('mutate', [
    lambda req, reg: req['submissions'][0]['payload']['annotations'][0]['judgment'].update(rationale='altered'),
    lambda req, reg: req['submissions'][0].update(role='adjudicator'),
    lambda req, reg: req['submissions'][0].update(signed_at='2027-01-01T00:00:00+00:00'),
    lambda req, reg: reg['reviewers'][0].update(revoked=True),
    lambda req, reg: reg['reviewers'][0].update(valid_until='2026-10-26'),
    lambda req, reg: reg['reviewers'][0].update(jurisdictions=['US']),
    lambda req, reg: reg['reviewers'][0].update(independence_group='generator-team'),
    lambda req, reg: reg['reviewers'][0].update(independence_group=reg['reviewers'][1]['independence_group']),
    lambda req, reg: reg['reviewers'][0].update(public_key=reg['reviewers'][1]['public_key']),
])
def test_tampered_unqualified_revoked_out_of_scope_nonindependent_records_rejected(signed, mutate):
    request, registry = signed
    mutate(request, registry)
    with pytest.raises(ValueError):
        check(request, registry)


def test_external_trust_pins_cannot_be_replaced_with_embedded_hashes(signed):
    request, registry = signed
    with pytest.raises(ValueError, match='registry'):
        assess_signed_annotations(request, registry=registry, expected_registry_sha256='0' * 64,
                                  expected_packet_sha256=request['packet']['packet_sha256'], now=NOW)
    with pytest.raises(ValueError, match='frozen'):
        assess_signed_annotations(request, registry=registry, expected_registry_sha256=digest(canonical_json(registry)),
                                  expected_packet_sha256='0' * 64, now=NOW)


def test_determinate_judgment_needs_exact_normative_evidence(signed, tmp_path):
    request, registry = signed
    payload = request['submissions'][0]['payload']
    payload['annotations'][0]['judgment']['labels']['faithfulness'] = 'supported'
    def reseal():
        request['submissions'][0] = sign_judgment(payload, packet_sha256=request['packet']['packet_sha256'],
                                                role='annotator', private_key_file=tmp_path / 'test-0.key', now=NOW)
    reseal()
    with pytest.raises(ValueError, match='normative source evidence'):
        check(request, registry)
    doc = request['packet']['units'][0]['family']['documents'][0]
    payload['annotations'][0]['judgment']['evidence_refs'] = [
        {'document_id': doc['id'], 'start': 0, 'end': len(doc['text']), 'quote': doc['text']}]
    reseal()
    assert check(request, registry)['status'] == 'adjudication_required'
    payload['annotations'][0]['judgment']['evidence_refs'][0]['quote'] = 'wrong quote'
    reseal()
    with pytest.raises(ValueError, match='source span'):
        check(request, registry)


def test_signed_third_party_adjudication_binds_original_submissions(signed, tmp_path):
    request, registry = signed
    payload = deepcopy(request['submissions'][1]['payload'])
    payload['annotations'][0]['judgment']['alternative_interpretations'] = ['An alternative test reading.']
    request['submissions'][1] = sign_judgment(payload, packet_sha256=request['packet']['packet_sha256'], role='annotator',
                                            private_key_file=tmp_path / 'test-1.key', now=NOW)
    assert check(request, registry)['status'] == 'adjudication_required'
    public = create_reviewer_key(tmp_path / 'third.key', tmp_path / 'third.pub')['public_key']
    registry['reviewers'].append({**registry['reviewers'][0], 'id': 'third', 'public_key': public,
                                  'roles': ['adjudicator'], 'independence_group': 'third-test-group'})
    adjudication = {'reviewer': reviewer('third'),
                    'submission_sha256s': [digest(canonical_json(s['payload'])) for s in request['submissions']],
                    'decisions': [{'unit_id': 'case-1', 'resolution': 'unresolved',
                                   'judgment': deepcopy(payload['annotations'][0]['judgment'])}]}
    request['adjudication'] = sign_judgment(adjudication, packet_sha256=request['packet']['packet_sha256'],
                                           role='adjudicator', private_key_file=tmp_path / 'third.key', now=NOW)
    assert check(request, registry)['unresolved_units'] == ['case-1']
    adjudication['decisions'][0]['resolution'] = 'resolved'
    request['adjudication'] = sign_judgment(adjudication, packet_sha256=request['packet']['packet_sha256'],
                                           role='adjudicator', private_key_file=tmp_path / 'third.key', now=NOW)
    assert check(request, registry)['status'] == 'annotations_collected'


def test_cli_verification_archives_public_registry_and_signed_evidence(signed, tmp_path):
    request, registry = signed
    source, roster, output = tmp_path / 'signed.json', tmp_path / 'roster.json', tmp_path / 'evidence.zip'
    source.write_text(json.dumps(request))
    roster.write_text(json.dumps(registry))
    completed = subprocess.run([sys.executable, '-m', 'src.cli', 'annotation-verify', '--input', str(source),
                                '--registry', str(roster), '--expected-registry-sha256', digest(canonical_json(registry)),
                                '--expected-packet-sha256', request['packet']['packet_sha256'], '--bundle', str(output)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result['key_signatures_verified']
    assert verify_bundle(output.read_bytes())['valid']
