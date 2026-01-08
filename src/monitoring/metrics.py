"""
Prometheus metrics for ML API monitoring

Tracks:
- Prediction count and latency
- Model confidence distribution
- Error rates
- Predictions by class
"""

import time
from functools import wraps
from typing import Callable

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, make_asgi_app

# =============================================================================
# METRICS DEFINITIONS
# =============================================================================

# Prediction metrics
prediction_total = Counter("brain_tumor_predictions_total", "Total number of predictions made", ["endpoint", "status"])

prediction_latency = Histogram(
    "brain_tumor_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

prediction_confidence = Histogram(
    "brain_tumor_prediction_confidence",
    "Model confidence scores",
    ["predicted_class"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0),
)

predictions_by_class = Counter("brain_tumor_predictions_by_class_total", "Total predictions per class", ["predicted_class"])

# Error metrics
error_total = Counter("brain_tumor_errors_total", "Total number of errors", ["error_type", "endpoint"])

# Model metrics
model_loaded = Gauge("brain_tumor_model_loaded", "Whether model is loaded (1=yes, 0=no)")

model_load_time = Gauge("brain_tumor_model_load_time_seconds", "Time taken to load model")

# System metrics
active_requests = Gauge("brain_tumor_active_requests", "Number of active requests being processed")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def track_prediction(predicted_class: str, confidence: float, latency: float, endpoint: str = "predict"):
    """
    Track a successful prediction

    Args:
        predicted_class: The predicted tumor class
        confidence: Model confidence (0-1)
        latency: Prediction latency in seconds
        endpoint: API endpoint name
    """
    prediction_total.labels(endpoint=endpoint, status="success").inc()
    prediction_latency.labels(endpoint=endpoint).observe(latency)
    prediction_confidence.labels(predicted_class=predicted_class).observe(confidence)
    predictions_by_class.labels(predicted_class=predicted_class).inc()


def track_error(error_type: str, endpoint: str = "predict"):
    """
    Track an error

    Args:
        error_type: Type of error (e.g., 'invalid_image', 'model_error')
        endpoint: API endpoint name
    """
    error_total.labels(error_type=error_type, endpoint=endpoint).inc()
    prediction_total.labels(endpoint=endpoint, status="error").inc()


def track_model_load(load_time: float, success: bool = True):
    """
    Track model loading

    Args:
        load_time: Time taken to load model in seconds
        success: Whether model loaded successfully
    """
    model_load_time.set(load_time)
    model_loaded.set(1 if success else 0)


def get_metrics_app():
    """
    Get ASGI app for Prometheus metrics endpoint

    Returns:
        ASGI application for /metrics endpoint
    """
    return make_asgi_app(registry=REGISTRY)


# =============================================================================
# DECORATOR FOR AUTOMATIC REQUEST TRACKING
# =============================================================================


def track_request(endpoint_name: str):
    """
    Decorator to automatically track request metrics

    Usage:
        @track_request("predict")
        async def predict_endpoint():
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                prediction_latency.labels(endpoint=endpoint_name).observe(latency)
                return result
            except Exception as e:
                track_error(error_type=type(e).__name__, endpoint=endpoint_name)
                raise
            finally:
                active_requests.dec()

        return wrapper

    return decorator
