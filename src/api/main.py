from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.schemas import PredictionResponse, HealthResponse
from src.config import MODELS_DIR, IMAGE_SIZE

# Initialize FastAPI
app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for classifying brain tumor MRI images using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
model = None
CLASS_NAMES = []


@app.on_event("startup")
async def load_model():
    """Load model and class names on startup"""
    global model, CLASS_NAMES

    model_path = MODELS_DIR / "brain_tumor_model.h5"
    class_names_path = MODELS_DIR / "class_names.json"

    # Load model
    if model_path.exists():
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️  Model not found at {model_path}")
        print(f"   Please train the model first: python src/training/train.py")

    # Load class names
    if class_names_path.exists():
        with open(class_names_path, 'r') as f:
            CLASS_NAMES = json.load(f)
        print(f"✅ Class names loaded: {CLASS_NAMES}")
    else:
        # Fallback to default class names
        CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
        print(f"⚠️  Class names file not found, using defaults: {CLASS_NAMES}")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns API status and model loading state
    """
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor type from MRI image

    **Parameters:**
    - **file**: MRI image file (JPG, PNG, or JPEG)

    **Returns:**
    - **predicted_class**: The predicted tumor type
    - **confidence**: Confidence score for the prediction (0-1)
    - **all_probabilities**: Probability scores for all classes

    **Example:**
    ```python
    import requests

    url = "http://localhost:8000/predict"
    files = {"file": open("mri_scan.jpg", "rb")}
    response = requests.post(url, files=files)
    print(response.json())
    ```
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize(IMAGE_SIZE)

        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # All probabilities
        all_probs = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, predictions)
        }

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/classes")
async def get_classes():
    """
    Get list of available tumor classes

    Returns all possible classification categories
    """
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


@app.get("/model-info")
async def get_model_info():
    """
    Get model information

    Returns details about the loaded model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_loaded": True,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_parameters": int(model.count_params()),
        "classes": CLASS_NAMES
    }