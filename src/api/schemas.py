from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for single image prediction"""

    predicted_class: str = Field(..., description="Predicted tumor class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probabilities for all classes"
    )
    inference_time_seconds: float = Field(
        ..., description="Time taken for inference in seconds"
    )
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "glioma",
                "confidence": 0.95,
                "all_probabilities": {
                    "glioma": 0.95,
                    "meningioma": 0.03,
                    "notumor": 0.01,
                    "pituitary": 0.01,
                },
                "inference_time_seconds": 0.0523,
                "timestamp": "2024-11-01T10:30:00",
            }
        }


class BatchPredictionItem(BaseModel):
    """Single prediction item in batch response"""

    filename: str = Field(..., description="Original filename")
    predicted_class: Optional[str] = Field(None, description="Predicted tumor class")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )
    all_probabilities: Optional[Dict[str, float]] = Field(
        None, description="Probabilities for all classes"
    )
    error: Optional[str] = Field(None, description="Error message if prediction failed")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""

    predictions: List[BatchPredictionItem] = Field(
        ..., description="List of predictions"
    )
    total_images: int = Field(..., description="Total number of images submitted")
    successful_predictions: int = Field(
        ..., description="Number of successful predictions"
    )
    total_time_seconds: float = Field(
        ..., description="Total processing time in seconds"
    )
    timestamp: str = Field(..., description="Batch prediction timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "filename": "scan1.jpg",
                        "predicted_class": "glioma",
                        "confidence": 0.95,
                        "all_probabilities": {
                            "glioma": 0.95,
                            "meningioma": 0.03,
                            "notumor": 0.01,
                            "pituitary": 0.01,
                        },
                        "error": None,
                    }
                ],
                "total_images": 1,
                "successful_predictions": 1,
                "total_time_seconds": 0.156,
                "timestamp": "2024-11-01T10:30:00",
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp (ISO format)")
    version: str = Field(..., description="API version")
    predictions_served: Optional[int] = Field(
        None, description="Total predictions served"
    )
    avg_inference_time: Optional[float] = Field(
        None, description="Average inference time in seconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-11-01T10:30:00",
                "version": "2.0.0",
                "predictions_served": 1523,
                "avg_inference_time": 0.052,
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information"""

    model_name: str = Field(..., description="Model architecture name")
    framework: str = Field(..., description="Deep learning framework used")
    input_shape: Any = Field(..., description="Model input shape")
    output_shape: Any = Field(..., description="Model output shape")
    total_parameters: int = Field(..., description="Total number of model parameters")
    classes: List[str] = Field(..., description="List of classification classes")
    image_size: tuple = Field(
        ..., description="Expected input image size (width, height)"
    )
    format: str = Field(..., description="Model file format (.keras or .h5)")
    loaded_at: str = Field(..., description="Model load timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "EfficientNetB0",
                "framework": "TensorFlow/Keras",
                "input_shape": [None, 224, 224, 3],
                "output_shape": [None, 4],
                "total_parameters": 4049564,
                "classes": ["glioma", "meningioma", "notumor", "pituitary"],
                "image_size": [224, 224],
                "format": "keras",
                "loaded_at": "2024-11-01T09:00:00",
            }
        }


class PredictionRequest(BaseModel):
    """Request model for prediction (if using JSON instead of file upload)"""

    image_base64: str = Field(..., description="Base64 encoded image")

    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""

    detail: str = Field(..., description="Error message")
    error: Optional[str] = Field(None, description="Detailed error information")
    path: Optional[str] = Field(None, description="Request path that caused the error")
    timestamp: Optional[str] = Field(None, description="Error timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Model not loaded",
                "error": "No model file found",
                "path": "/predict",
                "timestamp": "2024-11-01T10:30:00",
            }
        }


class StatisticsResponse(BaseModel):
    """Response model for API statistics"""

    total_predictions: int = Field(..., description="Total number of predictions made")
    total_inference_time: float = Field(
        ..., description="Total cumulative inference time"
    )
    average_inference_time: float = Field(
        ..., description="Average time per prediction"
    )
    model_loaded: bool = Field(..., description="Whether model is loaded")
    classes_available: int = Field(..., description="Number of classes available")

    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 1523,
                "total_inference_time": 79.2,
                "average_inference_time": 0.052,
                "model_loaded": True,
                "classes_available": 4,
            }
        }
