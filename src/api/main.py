import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.config import IMAGE_SIZE, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
model = None
CLASS_NAMES = []
MODEL_METADATA = {}

# Statistics tracking
PREDICTION_COUNT = 0
TOTAL_INFERENCE_TIME = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global model, CLASS_NAMES, MODEL_METADATA

    logger.info("🚀 Starting API server...")

    # Try loading .keras format first (recommended)
    keras_model_path = MODELS_DIR / "brain_tumor_model.keras"
    h5_model_path = MODELS_DIR / "brain_tumor_model.h5"
    class_names_path = MODELS_DIR / "class_names.json"

    # Load model
    model_loaded = False

    # Try .keras format first
    if keras_model_path.exists():
        try:
            start_time = time.time()
            model = tf.keras.models.load_model(keras_model_path)
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded from {keras_model_path} in {load_time:.2f}s")
            MODEL_METADATA["model_path"] = str(keras_model_path)
            MODEL_METADATA["format"] = "keras"
            model_loaded = True
        except Exception as e:
            logger.error(f"⚠️  Error loading .keras model: {e}")

    # Fallback to .h5 format
    if not model_loaded and h5_model_path.exists():
        try:
            start_time = time.time()
            model = tf.keras.models.load_model(h5_model_path)
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded from {h5_model_path} in {load_time:.2f}s")
            MODEL_METADATA["model_path"] = str(h5_model_path)
            MODEL_METADATA["format"] = "h5"
            model_loaded = True
        except Exception as e:
            logger.error(f"⚠️  Error loading .h5 model: {e}")

    if not model_loaded:
        logger.error(f"❌ No model found at {MODELS_DIR}")
        logger.error("   Please train the model first: python src/training/train.py")
    else:
        # Store model metadata
        MODEL_METADATA["input_shape"] = model.input_shape
        MODEL_METADATA["output_shape"] = model.output_shape
        MODEL_METADATA["total_params"] = model.count_params()
        MODEL_METADATA["loaded_at"] = datetime.now().isoformat()

    # Load class names
    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            CLASS_NAMES = json.load(f)
        logger.info(f"✅ Class names loaded: {CLASS_NAMES}")
        MODEL_METADATA["classes"] = CLASS_NAMES
    else:
        logger.warning(f"⚠️  Class names not found at {class_names_path}")
        CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
        MODEL_METADATA["classes"] = CLASS_NAMES

    logger.info("✨ API server ready!")

    yield

    # Shutdown
    logger.info("👋 Shutting down API server...")
    logger.info(f"📊 Total predictions served: {PREDICTION_COUNT}")
    if PREDICTION_COUNT > 0:
        avg_time = TOTAL_INFERENCE_TIME / PREDICTION_COUNT
        logger.info(f"⏱️  Average inference time: {avg_time:.3f}s")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Brain Tumor Classification API",
    description="Production-ready API for classifying brain tumor MRI images using deep learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "API Support",
        "email": "support@braintumor-api.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and response times"""
    start_time = time.time()

    # Log request - handle None client in test environment
    client_host = request.client.host if request.client else "test-client"
    logger.info(
        f"📥 {request.method} {request.url.path} - Client: {client_host}"
    )

    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log response
    logger.info(
        f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s"
    )

    return response


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns API status and model loading state
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def detailed_health():
    """
    Detailed health check with statistics

    Returns comprehensive health information including prediction statistics
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "predictions_served": PREDICTION_COUNT,
        "avg_inference_time": (
            round(TOTAL_INFERENCE_TIME / PREDICTION_COUNT, 3)
            if PREDICTION_COUNT > 0
            else 0
        ),
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get detailed model information

    Returns model architecture, parameters, and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": "EfficientNetB0",
        "framework": "TensorFlow/Keras",
        "input_shape": MODEL_METADATA.get("input_shape"),
        "output_shape": MODEL_METADATA.get("output_shape"),
        "total_parameters": MODEL_METADATA.get("total_params"),
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "format": MODEL_METADATA.get("format", "unknown"),
        "loaded_at": MODEL_METADATA.get("loaded_at"),
    }


@app.get("/classes", tags=["Model"])
async def get_classes():
    """
    Get available tumor classification classes

    Returns list of classes the model can predict
    """
    return {
        "classes": CLASS_NAMES,
        "count": len(CLASS_NAMES),
        "description": {
            "glioma": "A type of tumor that occurs in the brain and spinal cord",
            "meningioma": "Tumor that arises from the meninges",
            "pituitary": "Tumor in the pituitary gland",
            "notumor": "No tumor detected",
        },
    }


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model prediction

    Args:
        image: PIL Image object

    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Resize
    image = image.resize(IMAGE_SIZE)

    # Convert to array
    img_array = np.array(image)

    # EfficientNet preprocessing
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_array = preprocess_input(img_array.astype("float32"))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict tumor type from a single MRI image

    - **file**: MRI image file (JPEG, PNG)

    Returns prediction with confidence scores for all classes
    """
    global PREDICTION_COUNT, TOTAL_INFERENCE_TIME

    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please contact administrator."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file.",
        )

    try:
        # Read and preprocess image
        start_time = time.time()
        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        img_array = preprocess_image(image)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Calculate inference time
        inference_time = time.time() - start_time

        # Update statistics
        PREDICTION_COUNT += 1
        TOTAL_INFERENCE_TIME += inference_time

        # Get predicted class name
        predicted_class = CLASS_NAMES[predicted_class_idx]

        # Create probability dictionary
        all_probs = {
            CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
        }

        # Log prediction
        logger.info(
            f"🎯 Prediction: {predicted_class} (confidence: {confidence:.2%}) - Time: {inference_time:.3f}s"
        )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "inference_time_seconds": round(inference_time, 4),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict tumor type from multiple MRI images

    - **files**: List of MRI image files (JPEG, PNG)
    - Maximum 10 images per request

    Returns predictions for all images
    """
    global PREDICTION_COUNT, TOTAL_INFERENCE_TIME

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Limit batch size
    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 images per batch request"
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    predictions_list = []
    start_time = time.time()

    for idx, file in enumerate(files):
        # Validate file type
        if not file.content_type.startswith("image/"):
            predictions_list.append(
                {
                    "filename": file.filename,
                    "error": f"Invalid file type: {file.content_type}",
                    "predicted_class": None,
                    "confidence": None,
                    "all_probabilities": None,
                }
            )
            continue

        try:
            # Read and preprocess image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img_array = preprocess_image(image)

            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])

            # Get predicted class name
            predicted_class = CLASS_NAMES[predicted_class_idx]

            # Create probability dictionary
            all_probs = {
                CLASS_NAMES[i]: float(predictions[0][i])
                for i in range(len(CLASS_NAMES))
            }

            predictions_list.append(
                {
                    "filename": file.filename,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "all_probabilities": all_probs,
                    "error": None,
                }
            )

            PREDICTION_COUNT += 1

        except Exception as e:
            logger.error(f"❌ Error processing {file.filename}: {str(e)}")
            predictions_list.append(
                {
                    "filename": file.filename,
                    "error": str(e),
                    "predicted_class": None,
                    "confidence": None,
                    "all_probabilities": None,
                }
            )

    total_time = time.time() - start_time
    TOTAL_INFERENCE_TIME += total_time

    successful = sum(1 for p in predictions_list if p["error"] is None)

    logger.info(
        f"📦 Batch prediction: {successful}/{len(files)} successful - Time: {total_time:.3f}s"
    )

    return {
        "predictions": predictions_list,
        "total_images": len(files),
        "successful_predictions": successful,
        "total_time_seconds": round(total_time, 4),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """
    Get API usage statistics

    Returns current API usage metrics
    """
    return {
        "total_predictions": PREDICTION_COUNT,
        "total_inference_time": round(TOTAL_INFERENCE_TIME, 2),
        "average_inference_time": (
            round(TOTAL_INFERENCE_TIME / PREDICTION_COUNT, 4)
            if PREDICTION_COUNT > 0
            else 0
        ),
        "model_loaded": model is not None,
        "classes_available": len(CLASS_NAMES),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"❌ Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": str(request.url),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")