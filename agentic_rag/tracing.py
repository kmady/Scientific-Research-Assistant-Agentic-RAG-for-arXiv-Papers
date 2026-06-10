import os
import logging

logger = logging.getLogger(__name__)

_tracer = None


def setup_tracing(service_name: str = "agentic_rag") -> None:
    global _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        # Prefer Jaeger exporter if endpoint configured
        jaeger_endpoint = os.getenv("OTEL_JAEGER_ENDPOINT")
        if jaeger_endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                jaeger_exporter = JaegerExporter(agent_host_name=jaeger_endpoint.split(":")[0], agent_port=int(jaeger_endpoint.split(":")[1]))
                exporter = jaeger_exporter
            except Exception:
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry tracing initialized")
    except Exception as e:
        logger.info("OpenTelemetry not available or failed to initialize: %s", e)
        _tracer = None


def get_tracer():
    return _tracer


from contextlib import contextmanager

@contextmanager
def start_span(name: str):
    tracer = get_tracer()
    if not tracer:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        yield span
