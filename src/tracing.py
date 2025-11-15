"""
OpenTelemetry tracing configuration for Google Cloud Trace.
"""

import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

logger = logging.getLogger(__name__)


def setup_tracing(app):
    """
    Configure OpenTelemetry tracing with Google Cloud Trace export.
    
    Args:
        app: FastAPI application instance
    """
    try:
        # Get GCP project ID from environment
        project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        
        if not project_id:
            logger.warning("GCP_PROJECT_ID not set. Tracing will be disabled.")
            return
        
        # Create resource with service information
        resource = Resource.create({
            SERVICE_NAME: "iris-api",
            SERVICE_VERSION: "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
            "gcp.project_id": project_id
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Configure Cloud Trace exporter
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=project_id
        )
        
        # Add batch span processor
        tracer_provider.add_span_processor(
            BatchSpanProcessor(cloud_trace_exporter)
        )
        
        # Set the global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument FastAPI automatically
        FastAPIInstrumentor.instrument_app(app)
        
        logger.info(f"Tracing configured successfully for project: {project_id}")
        logger.info(f"View traces at: https://console.cloud.google.com/traces/list?project={project_id}")
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        logger.warning("Application will continue without tracing")


def get_tracer(name: str = "iris-api"):
    """
    Get a tracer instance for manual instrumentation.
    
    Args:
        name: Name of the tracer
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def trace_function(span_name: str):
    """
    Decorator to trace a function execution.
    
    Usage:
        @trace_function("my_function")
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator