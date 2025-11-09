"""
Gradio Demo App for Brain Tumor Classification

Usage:
    python src/demo/gradio_app.py

Access at: http://localhost:7860
"""

import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import IMAGE_SIZE, MODELS_DIR


class BrainTumorClassifier:
    """Brain Tumor Classifier for Gradio Demo"""

    def __init__(self):
        self.model = None
        self.class_names = []
        self.load_model()

    def load_model(self):
        """Load the trained model and class names"""
        # Try to load model
        keras_model = MODELS_DIR / "brain_tumor_model.keras"
        h5_model = MODELS_DIR / "brain_tumor_model.h5"

        if keras_model.exists():
            self.model = tf.keras.models.load_model(str(keras_model))
            print(f"✅ Model loaded from {keras_model}")
        elif h5_model.exists():
            self.model = tf.keras.models.load_model(str(h5_model))
            print(f"✅ Model loaded from {h5_model}")
        else:
            print("❌ No model found! Please train the model first.")

        # Load class names
        class_names_file = MODELS_DIR / "class_names.json"
        if class_names_file.exists():
            with open(class_names_file, "r") as f:
                self.class_names = json.load(f)
        else:
            self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize to expected size
        image = image.resize(IMAGE_SIZE)

        # Convert to array
        img_array = np.array(image)

        # Apply EfficientNet preprocessing
        from tensorflow.keras.applications.efficientnet import preprocess_input

        img_array = preprocess_input(img_array.astype("float32"))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return {
                "error": "Model not loaded. Please train the model first.",
                "prediction": None,
                "confidence": None,
                "all_probabilities": None,
            }

        # Preprocess
        img_array = self.preprocess_image(image)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])

        # Get class name
        predicted_class = self.class_names[predicted_idx]

        # All probabilities
        all_probs = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}

        return {"prediction": predicted_class, "confidence": confidence, "all_probabilities": all_probs}


# Initialize classifier
classifier = BrainTumorClassifier()


def classify_image(image):
    """
    Classify brain tumor from MRI image

    Args:
        image: PIL Image

    Returns:
        tuple: (predicted_class, confidence_dict, interpretation_text)
    """
    if image is None:
        return "Please upload an image", {}, ""

    # Get prediction
    result = classifier.predict(image)

    if "error" in result:
        return result["error"], {}, ""

    # Format output
    predicted_class = result["prediction"]
    confidence = result["confidence"]
    all_probs = result["all_probabilities"]

    # Create interpretation text
    interpretation = f"""
    ## 🎯 Prediction: **{predicted_class.upper()}**

    ### Confidence: **{confidence:.1%}**

    ### Class Descriptions:
    - **Glioma**: A tumor that occurs in the brain and spinal cord
    - **Meningioma**: A tumor that arises from the meninges (membranes surrounding brain/spinal cord)
    - **Pituitary**: A tumor in the pituitary gland at the base of the brain
    - **No Tumor**: No tumor detected in the MRI scan

    ### ⚠️ Disclaimer
    This is a demonstration model for educational purposes only. 
    Always consult qualified medical professionals for diagnosis.
    """

    return predicted_class, all_probs, interpretation


def create_demo():
    """Create Gradio interface"""

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-class {
        font-size: 1.5em;
        font-weight: bold;
        color: #2563eb;
    }
    """

    # Create interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Brain Tumor Classifier") as demo:
        gr.Markdown(
            """
        # 🧠 Brain Tumor Classification

        Upload an MRI scan to classify the type of brain tumor using deep learning.

        **Supported Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                image_input = gr.Image(type="pil", label="Upload MRI Scan", sources=["upload", "clipboard"], height=400)

                # Buttons
                with gr.Row():
                    predict_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([image_input], value="🗑️ Clear")

                # Examples
                gr.Examples(
                    examples=[
                        # Add paths to sample images if you have them
                        # ["examples/glioma_sample.jpg"],
                        # ["examples/meningioma_sample.jpg"],
                    ],
                    inputs=image_input,
                    label="Example Images (if available)",
                )

            with gr.Column(scale=1):
                # Outputs
                predicted_class = gr.Textbox(label="Predicted Class", elem_classes="output-class")

                confidence_plot = gr.Label(label="Confidence Scores", num_top_classes=4)

                interpretation = gr.Markdown(label="Interpretation")

        # Model info
        gr.Markdown(
            """
        ---
        ### 📊 Model Information
        - **Architecture:** EfficientNetB0 (Transfer Learning)
        - **Input Size:** 224x224 pixels
        - **Framework:** TensorFlow/Keras
        - **Training Data:** Brain MRI Images Dataset

        ### 🔧 Technical Details
        - Preprocessing: EfficientNet standard preprocessing
        - Output: 4-class classification (softmax)
        - Inference Time: ~50-100ms per image

        ### 📝 Notes
        - This model is trained for educational and demonstration purposes
        - Not intended for clinical diagnosis
        - Always consult medical professionals for health concerns
        """
        )

        # Connect components
        predict_btn.click(fn=classify_image, inputs=[image_input], outputs=[predicted_class, confidence_plot, interpretation])

        # Auto-predict on image upload
        image_input.change(fn=classify_image, inputs=[image_input], outputs=[predicted_class, confidence_plot, interpretation])

    return demo


if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()

    # Launch with options
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
        show_api=False,
    )
