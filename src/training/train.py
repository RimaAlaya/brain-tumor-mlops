import mlflow
import mlflow.keras
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cnn_model import (
    create_cnn_model,
    create_transfer_learning_model,
    compile_model,
    get_callbacks
)
from src.config import RAW_DATA_DIR, MODELS_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE


def count_samples_per_class(directory):
    """Count samples in each class"""
    from collections import defaultdict
    class_counts = defaultdict(int)

    for class_dir in Path(directory).iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*')))
            class_counts[class_dir.name] = count

    return dict(class_counts)


def calculate_class_weights(train_dir):
    """Calculate class weights to handle imbalance"""
    from sklearn.utils.class_weight import compute_class_weight

    class_counts = count_samples_per_class(train_dir)
    total = sum(class_counts.values())

    # Get class names sorted
    class_names = sorted(class_counts.keys())
    counts = [class_counts[name] for name in class_names]

    # Calculate weights
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.arange(len(class_names)),
        y=np.repeat(np.arange(len(class_names)), counts)
    )

    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    print("\n📊 Class Distribution:")
    for name, count in class_counts.items():
        print(f"   {name}: {count} samples ({count / total * 100:.1f}%)")

    print("\n⚖️  Class Weights (to handle imbalance):")
    for i, (name, weight) in enumerate(zip(class_names, class_weights_array)):
        print(f"   {name}: {weight:.3f}")

    return class_weights, class_names


def load_dataset(train_dir: Path, test_dir: Path, image_size=(224, 224), batch_size=16):
    """
    Load training and testing datasets with proper handling
    """

    print("\n📂 Loading dataset...")

    # Load training data with validation split
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    # Load testing data
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    class_names = train_ds.class_names

    # Normalize images to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def plot_training_history(history, save_path):
    """Plot training history"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Training plot saved to: {save_path}")


def evaluate_model(model, test_ds, class_names):
    """Detailed model evaluation"""

    print("\n" + "=" * 60)
    print("📊 DETAILED EVALUATION")
    print("=" * 60)

    # Get predictions
    y_true = []
    y_pred = []

    print("Running predictions on test set...")
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    # Calculate metrics per class
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {class_accuracy[i] * 100:.2f}%")

    return y_true, y_pred, cm


def train_model(train_dir: Path, test_dir: Path, model_save_path: Path, use_transfer_learning=False):
    """
    Train brain tumor classification model with MLflow tracking
    """

    print("=" * 60)
    print("🚀 Starting Brain Tumor Classification Training")
    print("=" * 60)

    # Calculate class weights
    class_weights, _ = calculate_class_weights(train_dir)

    # Set MLflow experiment
    mlflow.set_experiment("brain-tumor-classification")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("use_transfer_learning", use_transfer_learning)
        mlflow.log_param("class_weights", str(class_weights))

        # Load data
        train_ds, val_ds, test_ds, class_names = load_dataset(
            train_dir=train_dir,
            test_dir=test_dir,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )

        num_classes = len(class_names)
        mlflow.log_param("num_classes", num_classes)

        print(f"✅ Dataset loaded successfully!")
        print(f"   Classes: {class_names}")
        print(f"   Number of classes: {num_classes}")

        # Create model
        print(f"\n🏗️  Creating {'Transfer Learning' if use_transfer_learning else 'CNN'} model...")

        if use_transfer_learning:
            model = create_transfer_learning_model(
                input_shape=(*IMAGE_SIZE, 3),
                num_classes=num_classes,
                dropout_rate=0.5
            )
        else:
            model = create_cnn_model(
                input_shape=(*IMAGE_SIZE, 3),
                num_classes=num_classes,
                dropout_rate=0.5
            )

        model = compile_model(model, learning_rate=LEARNING_RATE)

        print("✅ Model created successfully!")
        print(f"   Total parameters: {model.count_params():,}")

        # Callbacks
        model_save_path.mkdir(parents=True, exist_ok=True)
        callbacks = get_callbacks(
            model_save_path=model_save_path / "best_model.h5",
            patience=10  # Increased patience
        )

        # Train
        print("\n🎯 Training model...")
        print("-" * 60)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,  # THIS IS CRITICAL
            verbose=1
        )

        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print("=" * 60)

        # Plot training history
        plot_path = model_save_path / "training_history.png"
        plot_training_history(history, plot_path)
        mlflow.log_artifact(plot_path)

        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)

        # Detailed evaluation
        y_true, y_pred, cm = evaluate_model(model, test_ds, class_names)

        # Log all metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_loss", test_metrics['loss'])

        # Calculate overfitting score
        overfitting_score = final_train_acc - final_val_acc
        mlflow.log_metric("overfitting_score", overfitting_score)

        # Save final model
        final_model_path = model_save_path / "brain_tumor_model.h5"
        model.save(final_model_path)

        # Log model to MLflow
        mlflow.keras.log_model(model, "model")

        # Save class names
        import json
        class_names_path = model_save_path / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)

        print("\n" + "=" * 60)
        print("📈 FINAL RESULTS")
        print("=" * 60)
        print(f"Training Accuracy:   {final_train_acc * 100:.2f}%")
        print(f"Validation Accuracy: {final_val_acc * 100:.2f}%")
        print(f"Test Accuracy:       {test_metrics['accuracy'] * 100:.2f}%")
        print(f"\n⚠️  Overfitting Score: {overfitting_score * 100:.2f}%")
        print(f"    (Lower is better, <10% is good)")
        print(f"\n💾 Model saved to: {final_model_path}")
        print(f"📝 Class names saved to: {class_names_path}")
        print("=" * 60)

        # Recommendations
        if test_metrics['accuracy'] < 0.5:
            print("\n⚠️  WARNING: Model accuracy is very low!")
            print("   Recommendation: Try transfer learning (option 2)")
        elif overfitting_score > 0.15:
            print("\n💡 Model is overfitting.")
            print("   Try: More data augmentation or transfer learning")
        elif test_metrics['accuracy'] > 0.80:
            print("\n✨ Excellent! Model is ready for deployment.")
        else:
            print("\n👍 Model performance is acceptable.")

        return model, history, class_names


if __name__ == "__main__":

    # Define data paths
    train_data_path = RAW_DATA_DIR / "Training"
    test_data_path = RAW_DATA_DIR / "Testing"

    # Check if paths exist
    if not train_data_path.exists():
        print(f"❌ ERROR: Training data not found at {train_data_path}")
        sys.exit(1)

    if not test_data_path.exists():
        print(f"❌ ERROR: Testing data not found at {test_data_path}")
        sys.exit(1)

    print("\n🤔 Choose model type:")
    print("   1. Custom CNN (good for learning)")
    print("   2. Transfer Learning - EfficientNet (RECOMMENDED - better accuracy)")

    choice = input("\nEnter choice (1 or 2, default=2): ").strip()
    use_transfer = (choice != "1")  # Default to transfer learning

    if use_transfer:
        print("\n✅ Using Transfer Learning - this will give much better results!")
    else:
        print("\n✅ Using Custom CNN")

    # Train the model
    model, history, class_names = train_model(
        train_dir=train_data_path,
        test_dir=test_data_path,
        model_save_path=MODELS_DIR,
        use_transfer_learning=use_transfer
    )

    print("\n✨ All done! Next steps:")
    print("   1. View MLflow experiments: mlflow ui")
    print("   2. Test the API: python -m uvicorn src.api.main:app --reload")
    print("   3. Run tests: pytest tests/ -v")