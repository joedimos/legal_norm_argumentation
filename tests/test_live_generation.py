from __future__ import annotations

import json
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from src.live_generation import generate_candidate, ollama_chat
from tests.test_response_alignment import candidate, family


def proposal(discloses=False):
    text = 'I will disclose the records.' if discloses else 'I will withhold the records.'
    constraint = {'op': 'atom', 'name': 'response:disclose'}
    if not discloses:
        constraint = {'op': 'not', 'args': [constraint]}
    return {'text': text, 'language': 'en', 'action_atoms': ['response:disclose'], 'mappings': [
        {'id': 'proposed', 'constraints': [constraint], 'source_refs': [{'start': 0, 'end': len(text), 'quote': text}],
         'rationale': 'Model proposal requiring independent review.'}]}


def request():
    return {'schema': 'vle-live/v1', 'family': family(), 'prompt': 'How will you handle these records?', 'model': 'test-model'}


def transport(output, calls):
    def send(payload, **kwargs):
        calls.append((payload, kwargs))
        return {'done': True, 'message': {'content': json.dumps(output)}, 'eval_count': 25}
    return send


def test_live_generation_does_not_approve_itself_or_send_receipts():
    calls = []
    result = generate_candidate(request(), transport=transport(proposal(), calls))
    assert result['generation_calls'] == 1
    assert result['status'] == 'review_required'
    assert result['result']['analysis']['status'] == 'robust_satisfied'
    assert 'reviews' not in result['candidate']
    assert 'reviews' not in json.loads(calls[0][0]['messages'][1]['content'])['family']
    assert result['usage']['eval_count'] == 25
    assert not result['deployment_authorized']


def test_generation_pauses_on_unreviewed_source_and_history_and_budget():
    data, calls = request(), []
    data['family'].pop('reviews')
    assert generate_candidate(data, transport=transport(proposal(), calls))['generation_calls'] == 0
    data = request()
    data['history'] = [candidate(data['family'], 'unreviewed', True, reviewed=False)]
    assert generate_candidate(data, transport=transport(proposal(), calls))['generation_calls'] == 0
    data['history'] = [candidate(data['family'], 'violating', True)]
    data['max_cycles'] = 1
    assert generate_candidate(data, transport=transport(proposal(), calls))['status'] == 'revision_limit'
    assert calls == []


def test_reviewed_violation_is_revised_with_counterexample_but_success_stops():
    data, calls = request(), []
    data['history'] = [candidate(data['family'], 'bad', True)]
    result = generate_candidate(data, transport=transport(proposal(), calls))
    assert result['generation_calls'] == 1
    prompt = json.loads(calls[0][0]['messages'][1]['content'])
    assert prompt['feedback']['counterexample']
    assert 'reviews' not in prompt['previous_candidate']
    data['history'] = [candidate(data['family'], 'good', False)]
    assert generate_candidate(data, transport=transport(proposal(), []))['status'] == 'model_satisfied'


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(reviews={'behavior': 'approved'}),
    lambda p: p['mappings'][0]['constraints'].append({'op': 'atom', 'name': 'fact:authorized'}),
    lambda p: p['mappings'][0]['source_refs'][0].update(quote='invented quote'),
    lambda p: p.update(text=''),
])
def test_untrusted_model_outputs_rejected(mutate):
    p = proposal()
    mutate(p)
    result = generate_candidate(request(), transport=transport(p, []))
    assert result['status'] == 'generation_failed'
    assert result['candidate'] is None


def test_truncation_duplicates_and_repeated_text_fail_closed():
    for raw in [ {'done': False}, {'done': True, 'done_reason': 'length'},
                 {'done': True, 'message': {'content': '{"text":1,"text":2}'}} ]:
        assert generate_candidate(request(), transport=lambda *a, raw=raw, **k: raw)['status'] == 'generation_failed'
    data = request()
    data['history'] = [candidate(data['family'], 'bad', True)]
    assert generate_candidate(data, transport=transport(proposal(True), []))['status'] == 'stalled'


def test_actual_http_transport_nonstreaming_payload_redirect_and_body_cap():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'done': True, 'message': {'content': json.dumps(proposal())}}).encode())

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        data = request()
        data['endpoint'] = f'http://127.0.0.1:{server.server_port}/api/chat'
        assert generate_candidate(data)['status'] == 'review_required'
        assert seen[0]['stream'] is False
        assert seen[0]['options']['num_predict'] == 2048
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('endpoint', ['http://example.com/api/chat', 'https://user:secret@example.com/api/chat',
                                    'https://example.com/api/chat?key=secret', 'file:///api/chat'])
def test_transport_endpoint_policy(endpoint):
    with pytest.raises(ValueError, match='endpoint'):
        ollama_chat({}, endpoint=endpoint, timeout=1)


def test_request_does_not_mutate_family_and_rejects_invalid_budget():
    data = request()
    original = deepcopy(data)
    generate_candidate(data, transport=transport(proposal(), []))
    assert data == original
    data['max_cycles'] = True
    with pytest.raises(ValueError, match='max_cycles'):
        generate_candidate(data)


def test_transport_rejects_redirects_oversize_and_timeout_without_retry(monkeypatch):
    from src import live_generation
    events = []
    class Response:
        status = 302

        def read(self, limit, **kwargs):
            return b'x' * limit

        def close(self):
            events.append('close')
    response = Response()
    class Pool:
        def request(self, *args, **kwargs):
            assert kwargs['retries'] is False and kwargs['redirect'] is False
            events.append('request')
            return response

        def clear(self):
            events.append('clear')
    monkeypatch.setattr(live_generation.urllib3, 'PoolManager', Pool)
    with pytest.raises(ValueError, match='302'):
        ollama_chat({}, endpoint='http://localhost:11434/api/chat', timeout=1)
    response.status = 200
    with pytest.raises(ValueError, match='1 MiB'):
        ollama_chat({}, endpoint='http://localhost:11434/api/chat', timeout=1)
    assert events.count('request') == 2
    assert events.count('close') == 2 and events.count('clear') == 2


def test_live_base_cannot_promote_response_action_to_fact():
    data = request()
    data['family']['observations'] = [{'response:disclose': False}]
    with pytest.raises(ValueError, match='trusted observations'):
        generate_candidate(data)
