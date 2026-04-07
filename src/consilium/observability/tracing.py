"""OpenTelemetry tracing configuration for Consilium."""
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracing(
    service_name: str = "consilium",
    phoenix_endpoint: str = "http://localhost:4318/v1/traces",
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing with Arize Phoenix backend.

    Args:
        service_name: Service name for trace resource
        phoenix_endpoint: OTLP HTTP endpoint URL

    Returns:
        Configured tracer instance
    """
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

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
