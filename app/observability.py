import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.metrics import set_meter_provider, get_meter
from opentelemetry.trace import get_tracer
from opentelemetry._logs import set_logger_provider

# Global OpenTelemetry variables
tracer = None
prediction_count = None
prediction_latency = None

def init_opentelemetry():
    global tracer, prediction_count, prediction_latency

    # OpenTelemetry Setup
    resource = Resource.create(attributes={
        "service.name": "intent-classifier-app",
        "service.version": "1.0.0",
    })

    # Tracing
    trace_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(trace_provider)
    span_exporter = OTLPSpanExporter(endpoint="http://lgtm:4318/v1/traces")
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    tracer = get_tracer(__name__)

    # Metrics
    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint="http://lgtm:4318/v1/metrics"))
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    set_meter_provider(meter_provider)
    meter = get_meter(__name__)

    # Custom Metrics
    prediction_count = meter.create_counter(
        "prediction_count",
        description="Number of predictions made"
    )
    prediction_latency = meter.create_histogram(
        "prediction_latency_seconds",
        description="Latency of model predictions in seconds",
        unit="s"
    )

    # Logging
    log_provider = LoggerProvider(resource=resource)
    set_logger_provider(log_provider)
    log_exporter = OTLPLogExporter(endpoint="http://lgtm:4318/v1/logs")
    log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set root logger level to INFO

    # Add OpenTelemetry LoggingHandler to the root logger
    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=log_provider)
    root_logger.addHandler(otel_handler)

    # Add a StreamHandler to also print logs to console
    stream_handler = logging.StreamHandler()
    root_logger.addHandler(stream_handler)

    # Instrument Python's standard logging (this will hook into the root logger)
    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging.INFO)

    # Force OpenTelemetry SDK internal logging to stdout for debugging
    otel_sdk_logger = logging.getLogger("opentelemetry")
    otel_sdk_logger.setLevel(logging.DEBUG)
    otel_sdk_logger.addHandler(logging.StreamHandler())
    otel_sdk_logger.propagate = False # Prevent duplication if root logger also handles it