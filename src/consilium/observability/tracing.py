"""OpenTelemetry tracing configuration for Consilium."""
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased


def init_tracing(
    service_name: str = "consilium",
    phoenix_endpoint: str = "http://localhost:4318/v1/traces",
    sample_rate: float = 1.0,
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing with Arize Phoenix backend.

    Args:
        service_name: Service name for trace resource
        phoenix_endpoint: OTLP HTTP endpoint URL
        sample_rate: Fraction of traces to sample (0.0–1.0). At 1.0 all traces
            are recorded (development default). Use TraceIdRatioBased sampler for
            any value below 1.0.

    Returns:
        Configured tracer instance
    """
    resource = Resource.create({"service.name": service_name})

    sampler: Optional[TraceIdRatioBased] = (
        TraceIdRatioBased(sample_rate) if sample_rate < 1.0 else None
    )

    provider = TracerProvider(resource=resource, sampler=sampler)  # type: ignore[arg-type]

    otlp_exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)

    return trace.get_tracer(__name__)


def instrument_fastapi(app: object) -> None:
    """
    Instrument FastAPI app with automatic request tracing.

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
