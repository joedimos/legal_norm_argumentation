from __future__ import annotations

from copy import deepcopy

import pytest

from src.audit_bundle import canonical_json, digest
from src.legal_annotation import DIMENSIONS, assess_annotations, prepare_annotation
from tests.test_response_alignment import bind_candidate, candidate, family


def study():
    base = family()
    bound = bind_candidate(base, candidate(base, 'draft', False))
    return {'schema': 'vle-annotation-study/v1', 'study_id': 'synthetic-test-only', 'units': [
        {'id': 'case-1', 'source_group': 'synthetic-policy-lineage', 'split': 'test', 'prompt': 'Handle records.',
         'family': bound}]}


def reviewer(id):
    return {'id': id, 'qualification': 'Synthetic test identity; not a legal expert.', 'independent': True,
            'conflicts_disclosed': False}


def assessment():
    packet = prepare_annotation(study())
    submissions = []
    for id in ('reviewer-A', 'reviewer-B'):
        submissions.append({'reviewer': reviewer(id), 'packet_sha256': packet['packet_sha256'], 'annotations': [
            {'unit_id': 'case-1', 'unit_sha256': packet['units'][0]['unit_sha256'], 'judgment': {
                'labels': dict.fromkeys(DIMENSIONS, 'uncertain'), 'rationale': 'Synthetic regression judgment.',
                'alternative_interpretations': []}}]})
    return {'schema': 'vle-annotation-assessment/v1', 'packet': packet, 'submissions': submissions}


def test_blinding_frozen_packet_and_original_submissions_retained():
    data = assessment()
    assert 'reviews' not in data['packet']['units'][0]['family']
    result = assess_annotations(data)
    assert result['status'] == 'annotations_collected'
    assert result['submissions'] == data['submissions']
    assert result['metrics']['faithfulness']['agreement'] == 1
    assert result['metrics']['faithfulness']['cohen_kappa'] is None
    assert not result['reviewers_authenticated']
    assert not result['legal_certification']
    assert 'reviews' not in result


def test_disagreement_requires_third_reviewer_bound_to_originals():
    data = assessment()
    data['submissions'][1]['annotations'][0]['judgment']['labels']['faithfulness'] = 'unsupported'
    assert assess_annotations(data)['unresolved_units'] == ['case-1']
    assert assess_annotations(data)['metrics']['faithfulness']['cohen_kappa'] == 0
    data['adjudication'] = {'reviewer': reviewer('third-reviewer'),
                            'submission_sha256s': [digest(canonical_json(s)) for s in data['submissions']],
                            'decisions': [{'unit_id': 'case-1', 'resolution': 'resolved',
                                           'judgment': deepcopy(data['submissions'][0]['annotations'][0]['judgment'])}]}
    result = assess_annotations(data)
    assert result['status'] == 'annotations_collected'
    assert result['disagreements']['case-1'] == ['faithfulness']
    data['adjudication']['decisions'][0]['resolution'] = 'unresolved'
    assert assess_annotations(data)['status'] == 'adjudication_required'
    data['adjudication']['reviewer']['id'] = 'reviewer-A'
    with pytest.raises(ValueError, match='third'):
        assess_annotations(data)


def test_alternative_interpretations_not_collapsed_by_equal_labels():
    data = assessment()
    data['submissions'][1]['annotations'][0]['judgment']['alternative_interpretations'] = ['Different legal reading.']
    assert assess_annotations(data)['disagreements']['case-1'] == ['alternative_interpretations']


@pytest.mark.parametrize('mutate', [
    lambda d: d['submissions'][1]['reviewer'].update(id='reviewer-A'),
    lambda d: d['submissions'][1]['reviewer'].update(independent=False),
    lambda d: d['submissions'][1]['reviewer'].update(conflicts_disclosed=True),
    lambda d: d['submissions'][1].update(packet_sha256='stale'),
    lambda d: d['submissions'][1]['annotations'][0].update(unit_sha256='stale'),
    lambda d: d['submissions'][1]['annotations'][0]['judgment']['labels'].pop('grounding'),
    lambda d: d['packet']['units'][0].update(prompt='tampered'),
    lambda d: d['packet'].update(predictions={'case-1': True}),
])
def test_stale_incomplete_nonindependent_or_unblinded_data_rejected(mutate):
    data = assessment()
    mutate(data)
    with pytest.raises(ValueError):
        assess_annotations(data)


def test_source_group_cannot_leak_between_splits_and_unknown_unit_fields_rejected():
    data = study()
    other = deepcopy(data['units'][0])
    other.update(id='case-2', split='train')
    data['units'].append(other)
    with pytest.raises(ValueError, match='crosses'):
        prepare_annotation(data)
    data = study()
    data['units'][0]['solver_result'] = 'robust_satisfied'
    with pytest.raises(ValueError, match='unit requires'):
        prepare_annotation(data)


def test_renamed_group_does_not_hide_identical_source_leakage():
    data = study()
    other = deepcopy(data['units'][0])
    other.update(id='case-2', source_group='renamed-group', split='train')
    data['units'].append(other)
    with pytest.raises(ValueError, match='snapshot crosses'):
        prepare_annotation(data)


def test_cli_annotation_assessment_roundtrip_in_audit_bundle(tmp_path):
    import json
    import subprocess
    import sys

    from src.audit_bundle import verify_bundle
    source, bundle = tmp_path / 'assessment.json', tmp_path / 'assessment.zip'
    source.write_text(json.dumps(assessment()))
    completed = subprocess.run([sys.executable, '-m', 'src.cli', 'annotation-assess', '--input', str(source),
                                '--bundle', str(bundle)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output['status'] == 'annotations_collected'
    assert verify_bundle(bundle.read_bytes(), expected_manifest_digest=output['bundle_manifest_digest'])['valid']
