from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import Settings

_configured = False


def configure_telemetry(settings: Settings) -> None:
    global _configured
    if _configured:
        return
    resource = Resource.create({"service.name": settings.service_name, "service.version": settings.service_version})
    provider = TracerProvider(resource=resource)
    if settings.otlp_endpoint:
        headers = {}
        if settings.otlp_headers:
            for pair in settings.otlp_headers.split(","):
                key, _, value = pair.partition("=")
                if key.strip() and value:
                    headers[key.strip()] = value.strip()
        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, headers=headers or None)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True


@contextmanager
def request_span(name: str, attributes: dict[str, object] | None = None) -> Iterator[trace.Span]:
    tracer = trace.get_tracer("legal_norm_argumentation.service")
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span
