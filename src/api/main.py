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
            print(f"   Model type: {model.name if hasattr(model, 'name') else 'Unknown'}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

            # Try loading .keras format
            keras_model_path = MODELS_DIR / "brain_tumor_model.keras"
            if keras_model_path.exists():
                try:
                    model = tf.keras.models.load_model(keras_model_path)
                    print(f"✅ Model loaded from {keras_model_path}")
                except Exception as e2:
                    print(f"❌ Error loading .keras model: {e2}")