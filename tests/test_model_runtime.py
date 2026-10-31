from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from src.live_generation import generate_candidate
from src.model_runtime import ModelError, model_doctor, request_json, resolve_config, vllm_chat
from tests.test_live_generation import proposal, request


@contextmanager
def runtime(*, model='test-model', status=200, finish='stop', encoding=None):
    events = []
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body):
            self.send_response(status)
            if encoding:
                self.send_header('Content-Encoding', encoding)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            events.append(('GET', self.path, None))
            self.reply({'data': [{'id': model}], 'models': [{'name': model, 'digest': 'a' * 64}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            events.append(('POST', self.path, payload))
            self.reply({'model': model, 'choices': [{'finish_reason': finish,
                        'message': {'role': 'assistant', 'content': json.dumps(proposal())}}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 20}})

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', events
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_vllm_http_generation_and_non_generating_inventory_check():
    with runtime() as (base, events):
        config = {'schema': 'vle-model-doctor/v1', 'provider': 'vllm', 'model': 'test-model',
                  'endpoint': base + '/v1/chat/completions'}
        doctor = model_doctor(config)
        assert doctor['status'] == 'model_available' and doctor['generation_calls'] == 0
        assert not doctor['inference_verified']
        data = {**request(), **{k: v for k, v in config.items() if k != 'schema'}}
        generated = generate_candidate(data)
        assert generated['status'] == 'review_required'
        assert generated['result']['analysis']['status'] == 'robust_satisfied'
        assert generated['usage']['eval_count'] == 20
        assert events[0] == ('GET', '/v1/models', None)
        payload = events[1][2]
        assert payload['max_tokens'] == 2048 and payload['n'] == 1
        assert payload['response_format'] == {'type': 'json_object'}
        assert 'options' not in payload and 'reviews' not in payload['messages'][1]['content']


@pytest.mark.parametrize('finish', ['length', 'content_filter', 'tool_calls', None])
def test_incomplete_vllm_output_never_selected(finish):
    with runtime(finish=finish) as (base, _):
        data = {**request(), 'provider': 'vllm', 'endpoint': base + '/v1/chat/completions'}
        assert generate_candidate(data)['reason'] == 'incomplete_or_refused_completion'


def test_returned_model_mismatch_and_missing_requested_inventory():
    with runtime(model='unexpected') as (base, _):
        config = {'schema': 'vle-model-doctor/v1', 'provider': 'vllm', 'model': 'test-model',
                  'endpoint': base + '/v1/chat/completions'}
        assert model_doctor(config)['status'] == 'model_not_listed'
        data = {**request(), **{k: v for k, v in config.items() if k != 'schema'}}
        assert generate_candidate(data)['reason'] == 'returned_model_mismatch'


@pytest.mark.parametrize('status,reason', [(401, 'authentication_failed'), (429, 'rate_limited'),
                                          (500, 'provider_http_error')])
def test_sanitized_transport_failures_do_not_retry(status, reason):
    with runtime(status=status) as (base, events):
        result = model_doctor({'schema': 'vle-model-doctor/v1', 'model': 'test-model', 'endpoint': base + '/api/chat'})
        assert result['reason'] == reason and result['status'] == 'runtime_unavailable'
        assert len(events) == 1


def test_ollama_inventory_records_advertised_digest_and_env_defaults(monkeypatch):
    with runtime() as (base, _):
        monkeypatch.setenv('VLE_MODEL_NAME', 'test-model')
        monkeypatch.setenv('VLE_MODEL_PROVIDER', 'ollama')
        monkeypatch.setenv('VLE_MODEL_ENDPOINT', base + '/api/chat')
        result = model_doctor({'schema': 'vle-model-doctor/v1'})
        assert result['listed_model_digest'] == 'a' * 64
        assert result['model'] == 'test-model'


def test_timeouts_are_classified_without_sensitive_exception_text(monkeypatch):
    from src import model_runtime
    class Pool:
        def request(self, *args, **kwargs):
            raise model_runtime.urllib3.exceptions.ReadTimeoutError(None, 'secret', 'secret')

        def clear(self):
            pass
    monkeypatch.setattr(model_runtime, 'BoundedHTTPClient', lambda **kw: Pool())
    with pytest.raises(ModelError, match='transport_timeout') as error:
        request_json('GET', 'http://localhost:11434/api/tags', timeout=1)
    assert 'secret' not in str(error.value)


def test_provider_messages_cannot_trigger_tool_calls(monkeypatch):
    from src import model_runtime
    monkeypatch.setattr(model_runtime, 'request_json', lambda *a, **kw: {
        'model': 'test-model', 'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{}', 'tool_calls': [{'function': 'approve'}]}}]})
    with pytest.raises(ModelError, match='invalid_assistant_message'):
        vllm_chat({'model': 'test-model'}, endpoint='http://localhost:8000/v1/chat/completions', timeout=1)


def test_unexpected_compression_fails_closed():
    with runtime(encoding='gzip') as (base, _):
        result = model_doctor({'schema': 'vle-model-doctor/v1', 'model': 'test-model', 'endpoint': base + '/api/chat'})
        assert result['reason'] == 'unexpected_content_encoding'


@pytest.mark.parametrize('config', [ {'provider': 'unknown', 'model': 'x'}, {'provider': [], 'model': 'x'},
                                     {'model': 'x', 'endpoint': 'http://localhost:bad/api/chat'},
                                     {'model': 'x', 'endpoint': 'https://example.com/api/chat?secret=1'} ])
def test_bad_runtime_config_rejected_before_network(config):
    with pytest.raises(ValueError):
        resolve_config(config)
