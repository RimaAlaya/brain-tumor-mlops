import os
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.keras
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.utils import to_categorical

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
)
from src.models.cnn_model import build_effnet, get_callbacks

# Set seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


def load_images(base_train, base_test, labels, image_size):
    """
    Load images from directories (EXACT notebook logic)
    - Load Training folder
    - Load Testing folder (append to X, y)
    - BGR -> RGB conversion
    - Resize to image_size
    """
    X = []
    y = []

    # Load Training
    for lbl in labels:
        folder = os.path.join(base_train, lbl)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(folder, fn))  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(labels.index(lbl))

    # Load Testing (append to X, y like notebook)
    for lbl in labels:
        folder = os.path.join(base_test, lbl)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(folder, fn))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(labels.index(lbl))

    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.int32)

    print("Loaded images:", X.shape, "labels:", np.bincount(y))

    return X, y


def preprocess_and_split(X, y, labels, random_state):
    """
    Preprocess + split (EXACT notebook logic)
    - EfficientNet preprocess_input on whole array
    - shuffle
    - train_test_split 90/10
    - to_categorical
    """
    # EfficientNet preprocessing
    X_proc = eff_preprocess(X.astype("float32"))

    # Shuffle & split
    X_proc, y = shuffle(X_proc, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.10, random_state=random_state)

    y_train_cat = to_categorical(y_train, num_classes=len(labels))
    y_test_cat = to_categorical(y_test, num_classes=len(labels))

    print(
        "Train:",
        X_train.shape,
        y_train_cat.shape,
        "Test:",
        X_test.shape,
        y_test_cat.shape,
    )

    return X_train, X_test, y_train_cat, y_test_cat, y_test


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_and_plot(model, X_test, y_test, labels, save_dir):
    """
    Evaluate model (EXACT notebook logic)
    - classification_report
    - confusion matrix (normalized)
    - seaborn heatmap
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATION")
    print("=" * 60)

    preds = model.predict(X_test, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_test

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))

    # Confusion matrix (normalized)
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")

    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Confusion matrix saved: {cm_path}")

    return y_true, y_pred, cm


def train_model():
    """
    Main training function (EXACT notebook flow)
    """
    print("=" * 60)
    print("🧠 Brain Tumor Classification - EfficientNetB0")
    print("=" * 60)

    # Setup
    labels = ["glioma", "notumor", "meningioma", "pituitary"]
    base_train = str(RAW_DATA_DIR / "Training")
    base_test = str(RAW_DATA_DIR / "Testing")

    if not os.path.exists(base_train):
        print(f"❌ ERROR: Training data not found at {base_train}")
        sys.exit(1)

    if not os.path.exists(base_test):
        print(f"❌ ERROR: Testing data not found at {base_test}")
        sys.exit(1)

    # MLflow
    mlflow.set_experiment("brain-tumor-efficientnet-clean")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_params(
            {
                "image_size": IMAGE_SIZE[0],
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "random_state": RANDOM_STATE,
                "model": "EfficientNetB0",
                "trainable": "full_backbone",
                "dropout": 0.5,
            }
        )

        # Load images
        print("\n" + "=" * 60)
        print("STEP 1: Loading images...")
        print("=" * 60)
        t0 = time.time()
        X, y = load_images(base_train, base_test, labels, IMAGE_SIZE[0])
        print(f"Loading time: {time.time() - t0:.1f}s")

        # Preprocess & split
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing & splitting...")
        print("=" * 60)
        X_train, X_test, y_train_cat, y_test_cat, y_test = preprocess_and_split(X, y, labels, RANDOM_STATE)

        # Build model
        print("\n" + "=" * 60)
        print("STEP 3: Building model...")
        print("=" * 60)
        model = build_effnet(
            image_size=IMAGE_SIZE,
            num_classes=len(labels),
            dropout_rate=0.5,
            learning_rate=LEARNING_RATE,
        )

        print(f"\nTotal parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

        # Callbacks
        checkpoint_path = MODELS_DIR / "effnet_best.keras"
        callbacks = get_callbacks(checkpoint_path)

        # Train
        print("\n" + "=" * 60)
        print("STEP 4: Training...")
        print("=" * 60)
        t0 = time.time()

        history = model.fit(
            X_train,
            y_train_cat,
            validation_split=0.10,  # Same as notebook
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = time.time() - t0
        print(f"\n✅ Training time: {training_time:.1f}s")

        # Plot training history
        plot_path = MODELS_DIR / "training_history.png"
        plot_training_history(history, plot_path)
        mlflow.log_artifact(plot_path)

        # Load best model
        print("\n" + "=" * 60)
        print("STEP 5: Loading best model & evaluating...")
        print("=" * 60)
        best_model = tf.keras.models.load_model(str(checkpoint_path), compile=False)

        # Evaluate
        y_true, y_pred, cm = evaluate_and_plot(best_model, X_test, y_test, labels, MODELS_DIR)

        # Calculate metrics
        from sklearn.metrics import accuracy_score

        test_accuracy = accuracy_score(y_true, y_pred)

        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]

        # Log metrics
        mlflow.log_metrics(
            {
                "final_train_accuracy": final_train_acc,
                "final_val_accuracy": final_val_acc,
                "test_accuracy": test_accuracy,
                "training_time_seconds": training_time,
            }
        )

        # === FIX: Save using .keras format (native Keras 3 format) ===
        print("\n" + "=" * 60)
        print("STEP 6: Saving final model...")
        print("=" * 60)

        # Save in native Keras format (.keras) - RECOMMENDED
        final_model_path_keras = MODELS_DIR / "brain_tumor_model.keras"
        best_model.save(str(final_model_path_keras))
        print(f"✅ Model saved: {final_model_path_keras}")

        # Also save in .h5 format for backward compatibility (using save_weights instead)
        final_model_path_h5 = MODELS_DIR / "brain_tumor_model.h5"
        try:
            # Save only weights to avoid pickle issues
            best_model.save_weights(str(final_model_path_h5))
            print(f"✅ Model weights saved: {final_model_path_h5}")
        except Exception as e:
            print(f"⚠️  Could not save .h5 format: {e}")
            print("   (This is OK - using .keras format instead)")

        # Log model to MLflow
        mlflow.keras.log_model(best_model, "model")

        # Save class names
        import json

        class_names_path = MODELS_DIR / "class_names.json"
        # Convert labels to match API expectations
        api_labels = [lbl.replace("_tumor", "").replace("no_", "notumor") for lbl in labels]
        with open(class_names_path, "w") as f:
            json.dump(api_labels, f)

        # Final report
        print("\n" + "=" * 60)
        print("📈 FINAL RESULTS")
        print("=" * 60)
        print(f"Training Accuracy:   {final_train_acc * 100:.2f}%")
        print(f"Validation Accuracy: {final_val_acc * 100:.2f}%")
        print(f"Test Accuracy:       {test_accuracy * 100:.2f}%")
        print(f"\n💾 Model saved: {final_model_path_keras}")
        print(f"💾 Model weights: {final_model_path_h5}")
        print(f"📝 Class names: {class_names_path}")
        print(f"📊 Confusion matrix: {MODELS_DIR / 'confusion_matrix.png'}")
        print(f"📈 Training plot: {plot_path}")
        print("=" * 60)

        # Performance check
        if test_accuracy > 0.85:
            print("\n✨ Excellent! Model is ready for deployment.")
        elif test_accuracy > 0.75:
            print("\n👍 Good performance!")
        else:
            print("\n⚠️  Performance could be improved.")

        print("\n📌 README snippet:")
        print("- Backbone: EfficientNetB0 (end-to-end fine-tuned)")
        print("- Input: 224x224 RGB")
        print("- Preprocess: EfficientNet preprocess_input")
        print("- Training: validation_split=0.1, epochs=12, Adam lr=1e-3, ReduceLROnPlateau")
        print("- Metric: classification_report + confusion matrix")

        return best_model, history


if __name__ == "__main__":
    model, history = train_model()

    print("\n✨ Training complete!")
    print("\n📌 Next steps:")
    print("   1. View MLflow: mlflow ui")
    print("   2. Start API: uvicorn src.api.main:app --reload")
