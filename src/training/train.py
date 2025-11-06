"""
Enhanced Training Script with Production-Grade MLflow Integration

Key Improvements:
- Model registry integration
- Automated best model promotion
- Comprehensive experiment tracking
- System metrics logging
- Rich metadata tagging
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

from src.config import RAW_DATA_DIR, MODELS_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE
from src.models.cnn_model import build_effnet, get_callbacks

# Import enhanced MLflow utilities
from src.mlflow_utils import ModelRegistry, ExperimentTracker, ManagedRun

# Set seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


def load_images(base_train, base_test, labels, image_size):
    """Load images from training and testing directories"""
    X = []
    y = []

    # Load Training
    for lbl in labels:
        folder = os.path.join(base_train, lbl)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, fn))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(labels.index(lbl))

    # Load Testing
    for lbl in labels:
        folder = os.path.join(base_test, lbl)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, fn))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(labels.index(lbl))

    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.int32)

    print(f"✅ Loaded {len(X)} images")
    return X, y


def preprocess_and_split(X, y, labels, random_state):
    """Preprocess and split data"""
    X_proc = eff_preprocess(X.astype('float32'))
    X_proc, y = shuffle(X_proc, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.10, random_state=random_state
    )

    y_train_cat = to_categorical(y_train, num_classes=len(labels))
    y_test_cat = to_categorical(y_test, num_classes=len(labels))

    return X_train, X_test, y_train_cat, y_test_cat, y_test


def train_model():
    """
    Main training function with enhanced MLflow tracking
    """
    print("=" * 60)
    print("🧠 Brain Tumor Classification - Enhanced MLflow Training")
    print("=" * 60)

    # Setup
    labels = ['glioma', 'notumor', 'meningioma', 'pituitary']
    base_train = str(RAW_DATA_DIR / "Training")
    base_test = str(RAW_DATA_DIR / "Testing")

    if not os.path.exists(base_train) or not os.path.exists(base_test):
        print(f"❌ ERROR: Training or testing data not found")
        sys.exit(1)

    # Initialize enhanced MLflow utilities
    tracker = ExperimentTracker("brain-tumor-production")
    registry = ModelRegistry()

    # Start managed run (auto-ends on completion or error)
    with ManagedRun(
            tracker,
            run_name=f"efficientnet_b0_{int(time.time())}",
            model_architecture="EfficientNetB0",
            training_type="full_pipeline"
    ):
        # Log dataset info
        tracker.log_dataset_info({
            "train_dir": base_train,
            "test_dir": base_test,
            "num_classes": len(labels),
            "class_names": str(labels),
            "image_size": IMAGE_SIZE[0]
        })

        # Log hyperparameters
        tracker.log_params({
            "model": "EfficientNetB0",
            "image_size": IMAGE_SIZE[0],
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "random_state": RANDOM_STATE,
            "trainable_layers": "all",
            "dropout_rate": 0.5,
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "validation_split": 0.10
        })

        # Log code version (git commit)
        tracker.log_code_version()

        # Load images
        print("\n📂 Loading images...")
        t0 = time.time()
        X, y = load_images(base_train, base_test, labels, IMAGE_SIZE[0])
        load_time = time.time() - t0

        tracker.log_metrics({
            "data_loading_time_seconds": load_time,
            "total_images": len(X)
        })

        # Preprocess & split
        print("\n⚙️  Preprocessing data...")
        X_train, X_test, y_train_cat, y_test_cat, y_test = preprocess_and_split(
            X, y, labels, RANDOM_STATE
        )

        tracker.log_metrics({
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        })

        # Build model
        print("\n🏗️  Building model...")
        model = build_effnet(
            image_size=IMAGE_SIZE,
            num_classes=len(labels),
            dropout_rate=0.5,
            learning_rate=LEARNING_RATE
        )

        tracker.log_metrics({
            "total_parameters": model.count_params(),
            "trainable_parameters": sum([tf.size(w).numpy() for w in model.trainable_weights])
        })

        # Callbacks
        checkpoint_path = MODELS_DIR / "effnet_best.keras"
        callbacks = get_callbacks(checkpoint_path)

        # Train
        print("\n🏋️  Training model...")
        t0 = time.time()

        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.10,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - t0

        # Log training time
        tracker.log_metrics({
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60
        })

        # Log training curves
        tracker.log_training_curves(history)

        # Log epoch-wise metrics
        for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(
                history.history['accuracy'],
                history.history['val_accuracy'],
                history.history['loss'],
                history.history['val_loss']
        )):
            tracker.log_metrics({
                "train_accuracy": acc,
                "val_accuracy": val_acc,
                "train_loss": loss,
                "val_loss": val_loss
            }, step=epoch)

        # Load best model
        print("\n📦 Loading best model...")
        best_model = tf.keras.models.load_model(str(checkpoint_path), compile=False)

        # Evaluate
        print("\n📊 Evaluating model...")
        y_true = y_test
        preds = best_model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        # Calculate metrics
        test_accuracy = accuracy_score(y_true, y_pred)
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        # Log final metrics
        tracker.log_metrics({
            "final_train_accuracy": final_train_acc,
            "final_val_accuracy": final_val_acc,
            "test_accuracy": test_accuracy
        })

        # Log classification report
        tracker.log_classification_report(y_true, y_pred, labels)

        # Log confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tracker.log_confusion_matrix(cm, labels)

        # Save model artifacts
        print("\n💾 Saving model...")

        # Save in native Keras format
        final_model_path_keras = MODELS_DIR / "brain_tumor_model.keras"
        best_model.save(str(final_model_path_keras))

        # Log model to MLflow
        tracker.log_model(best_model, artifact_path="model")

        # Save class names
        import json
        class_names_path = MODELS_DIR / "class_names.json"
        api_labels = [lbl.replace('_tumor', '').replace('no_', 'notumor') for lbl in labels]
        with open(class_names_path, 'w') as f:
            json.dump(api_labels, f)

        tracker.log_artifact(str(class_names_path))

        # Get current run ID for model registry
        run_id = tracker.run.info.run_id

        # Register model in MLflow Model Registry
        print("\n📝 Registering model in MLflow Registry...")
        model_uri = f"runs:/{run_id}/model"

        registration = registry.register_model(
            model_uri=model_uri,
            model_name="brain_tumor_classifier",
            tags={
                "accuracy": f"{test_accuracy:.4f}",
                "architecture": "EfficientNetB0",
                "framework": "tensorflow",
                "dataset": "brain_mri",
                "image_size": str(IMAGE_SIZE[0])
            },
            description=f"Brain tumor classifier trained on {len(X_train)} samples with {test_accuracy:.2%} test accuracy"
        )

        print(f"✅ Model registered as v{registration['version']}")

        # Auto-promote if meets threshold
        print("\n🎯 Checking for auto-promotion...")
        promotion_threshold = 0.85  # 85% accuracy threshold

        if test_accuracy >= promotion_threshold:
            promotion = registry.auto_promote_best_model(
                model_name="brain_tumor_classifier",
                metric="test_accuracy",
                threshold=promotion_threshold,
                stage="Production"
            )

            if promotion:
                print(f"🚀 Model auto-promoted to Production!")
            else:
                print(f"⚠️  Model meets threshold but not promoted (check logs)")
        else:
            print(f"⚠️  Model accuracy ({test_accuracy:.2%}) below threshold ({promotion_threshold:.2%})")
            print(f"   Model registered but not promoted to Production")

        # Final report
        print("\n" + "=" * 60)
        print("📈 TRAINING COMPLETE")
        print("=" * 60)
        print(f"Training Accuracy:   {final_train_acc * 100:.2f}%")
        print(f"Validation Accuracy: {final_val_acc * 100:.2f}%")
        print(f"Test Accuracy:       {test_accuracy * 100:.2f}%")
        print(f"Training Time:       {training_time / 60:.1f} minutes")
        print(f"\n💾 Model saved: {final_model_path_keras}")
        print(f"📊 MLflow Run ID: {run_id}")
        print(f"🏷️  Model Registry: brain_tumor_classifier v{registration['version']}")
        print("=" * 60)

        return best_model, history, registration


if __name__ == "__main__":
    try:
        model, history, registration = train_model()

        print("\n✨ Training pipeline completed successfully!")
        print("\n📌 Next steps:")
        print("   1. View MLflow UI: mlflow ui")
        print("   2. Compare experiments in MLflow")
        print("   3. Check model registry for versions")
        print("   4. Start API: docker-compose up -d api")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)