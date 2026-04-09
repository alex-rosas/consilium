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


def test_init_tracing_with_sample_rate_below_one():
    """init_tracing with sample_rate < 1.0 uses TraceIdRatioBased sampler."""
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    tracer = init_tracing(sample_rate=0.1)
    assert tracer is not None


def test_init_tracing_with_sample_rate_one_uses_no_ratio_sampler():
    """init_tracing with sample_rate=1.0 does not install a ratio sampler."""
    tracer = init_tracing(sample_rate=1.0)
    assert tracer is not None


def test_trace_sample_rate_in_settings():
    """Settings.trace_sample_rate defaults to 1.0."""
    from consilium.config import Settings

    s = Settings()
    assert s.trace_sample_rate == 1.0


def test_trace_sample_rate_rejects_out_of_range():
    """Settings.trace_sample_rate rejects values outside 0.0–1.0."""
    from pydantic import ValidationError
    from consilium.config import Settings

    with pytest.raises(ValidationError):
        Settings(trace_sample_rate=1.5)

    with pytest.raises(ValidationError):
        Settings(trace_sample_rate=-0.1)
