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

    # Try loading .keras format first (recommended)
    keras_model_path = MODELS_DIR / "brain_tumor_model.keras"
    h5_model_path = MODELS_DIR / "brain_tumor_model.h5"
    class_names_path = MODELS_DIR / "class_names.json"

    # Load model
    model_loaded = False

    # Try .keras format first
    if keras_model_path.exists():
        try:
            model = tf.keras.models.load_model(keras_model_path)
            print(f"✅ Model loaded from {keras_model_path}")
            model_loaded = True
        except Exception as e:
            print(f"⚠️  Error loading .keras model: {e}")

    # Fallback to .h5 format
    if not model_loaded and h5_model_path.exists():
        try:
            model = tf.keras.models.load_model(h5_model_path)
            print(f"✅ Model loaded from {h5_model_path}")
            model_loaded = True
        except Exception as e:
            print(f"⚠️  Error loading .h5 model: {e}")

    if not model_loaded:
        print(f"❌ No model found at {MODELS_DIR}")
        print("   Please train the model first: python src/training/train.py")

    # Load class names
    if class_names_path.exists():
        with open(class_names_path, 'r') as f:
            CLASS_NAMES = json.load(f)
        print(f"✅ Class names loaded: {CLASS_NAMES}")
    else:
        print(f"⚠️  Class names not found at {class_names_path}")
        CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/classes")
async def get_classes():
    """Get available tumor classes"""
    return {"classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict tumor type from MRI image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        # Convert to array and preprocess (EfficientNet preprocessing)
        img_array = np.array(image)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array.astype('float32'))
        img_array = np.expand_dims(img_array, axis=0)

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

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)