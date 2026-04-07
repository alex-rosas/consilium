"""Unit tests for OpenTelemetry tracing."""
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from consilium.observability.tracing import init_tracing


@pytest.fixture(autouse=True)
def reset_tracer_provider():
    """Reset the global tracer provider to a no-op before each test."""
    trace.set_tracer_provider(TracerProvider())
    yield
    trace.set_tracer_provider(TracerProvider())


def test_init_tracing_returns_tracer():
    """init_tracing returns a valid tracer."""
    tracer = init_tracing()
    assert tracer is not None


def test_tracer_creates_spans():
    """Tracer can create spans with attributes."""
    tracer = init_tracing()

    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.key", "test.value")
        # Span created successfully — no exception means pass


def test_multiple_spans_in_sequence():
    """Multiple spans can be created in sequence."""
    tracer = init_tracing()

    with tracer.start_as_current_span("span-1") as span1:
        span1.set_attribute("span.id", 1)

    with tracer.start_as_current_span("span-2") as span2:
        span2.set_attribute("span.id", 2)

    # Both spans created without errors
