"""Bounded inference transports and non-generating runtime diagnostics."""
from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

import urllib3

from .audit_bundle import canonical_json, digest

MAX_RESPONSE = 1024 * 1024
PATHS = {"ollama": "/api/chat", "vllm": "/v1/chat/completions"}


class ModelError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def strict_json(raw):
    import json

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def validate_endpoint(endpoint, provider="ollama"):
    if not isinstance(endpoint, str) or provider not in PATHS:
        raise ValueError("invalid endpoint/provider")
    url = urlsplit(endpoint)
    if (url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path != PATHS[provider]
            or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("endpoint requires HTTPS or loopback HTTP, correct provider path, no credentials/query")
    # Force port validation before recording or making a request.
    _ = url.port


def resolve_config(request):
    provider = request.get("provider", os.environ.get("VLE_MODEL_PROVIDER", "ollama"))
    if not isinstance(provider, str) or provider not in PATHS:
        raise ValueError("provider must be ollama or vllm")
    model = request.get("model", os.environ.get("VLE_MODEL_NAME"))
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise ValueError("configure model or VLE_MODEL_NAME")
    default = "http://localhost:11434/api/chat" if provider == "ollama" else "http://localhost:8000/v1/chat/completions"
    endpoint = request.get("endpoint", os.environ.get("VLE_MODEL_ENDPOINT", default))
    validate_endpoint(endpoint, provider)
    timeout = request.get("timeout_seconds", 60)
    if type(timeout) is not int or not 1 <= timeout <= 120:
        raise ValueError("timeout_seconds must be 1-120")
    return {"provider": provider, "model": model, "endpoint": endpoint, "timeout_seconds": timeout}


def request_json(method, endpoint, *, timeout, payload=None, api_key=None):
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    pool = urllib3.PoolManager()
    try:
        response = pool.request(method, endpoint, body=canonical_json(payload) if payload is not None else None,
                                headers=headers, timeout=urllib3.Timeout(total=timeout), retries=False,
                                redirect=False, preload_content=False)
        try:
            if response.status != 200:
                raise ModelError({401: "authentication_failed", 403: "permission_denied", 404: "endpoint_or_model_missing",
                                  429: "rate_limited"}.get(response.status, "provider_http_error"))
            if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise ModelError("unexpected_content_encoding")
            raw = response.read(MAX_RESPONSE + 1, decode_content=False)
            if len(raw) > MAX_RESPONSE:
                raise ModelError("response_exceeds_1_MiB")
            try:
                value = strict_json(raw)
            except (ValueError, RecursionError) as exc:
                raise ModelError("invalid_provider_json") from exc
            if not isinstance(value, dict):
                raise ModelError("invalid_provider_envelope")
            return value
        finally:
            response.close()
    except urllib3.exceptions.NewConnectionError as exc:
        raise ModelError("runtime_unreachable") from exc
    except urllib3.exceptions.TimeoutError as exc:
        raise ModelError("transport_timeout") from exc
    except urllib3.exceptions.HTTPError as exc:
        raise ModelError("runtime_unreachable") from exc
    finally:
        pool.clear()


def ollama_chat(payload, *, endpoint, timeout, api_key=None):
    validate_endpoint(endpoint, "ollama")
    return request_json("POST", endpoint, timeout=timeout, payload=payload, api_key=api_key)


def vllm_chat(payload, *, endpoint, timeout, api_key=None):
    validate_endpoint(endpoint, "vllm")
    raw = request_json("POST", endpoint, timeout=timeout, payload=payload, api_key=api_key)
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ModelError("invalid_completion_choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ModelError("incomplete_or_refused_completion")
    message = choice.get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant" or message.get("tool_calls")
            or message.get("refusal") or not isinstance(message.get("content"), str)):
        raise ModelError("invalid_assistant_message")
    if raw.get("model") != payload["model"]:
        raise ModelError("returned_model_mismatch")
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return {"done": True, "done_reason": "stop", "message": message, "model": raw["model"],
            "prompt_eval_count": usage.get("prompt_tokens"), "eval_count": usage.get("completion_tokens"),
            "provider_response_sha256": digest(canonical_json(raw))}


def model_doctor(request):
    if (not isinstance(request, dict) or request.get("schema") != "vle-model-doctor/v1"
            or set(request) - {"schema", "provider", "model", "endpoint", "timeout_seconds"}):
        raise ValueError("invalid model diagnostic request")
    config = resolve_config(request)
    url = urlsplit(config["endpoint"])
    path = "/api/tags" if config["provider"] == "ollama" else "/v1/models"
    endpoint = urlunsplit((url.scheme, url.netloc, path, "", ""))
    result = {"schema": "vle-model-doctor-result/v1", **config, "generation_calls": 0,
              "inference_verified": False, "legal_certification": False}
    try:
        raw = request_json("GET", endpoint, timeout=config["timeout_seconds"], api_key=os.environ.get("VLE_MODEL_API_KEY"))
        models = raw.get("models" if config["provider"] == "ollama" else "data")
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise ModelError("invalid_model_inventory")
        key = "name" if config["provider"] == "ollama" else "id"
        matched = [item for item in models if item.get(key) == config["model"]]
        return {**result, "status": "model_available" if len(matched) == 1 else "model_not_listed",
                "inventory_sha256": digest(canonical_json(raw)),
                "listed_model_digest": matched[0].get("digest") if len(matched) == 1 else None}
    except ModelError as exc:
        return {**result, "status": "runtime_unavailable", "reason": exc.code}
