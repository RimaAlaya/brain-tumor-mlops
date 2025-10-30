import mlflow
import mlflow.keras
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cnn_model import (
    create_efficientnet_model,
    compile_model,
    unfreeze_model
)
from src.config import RAW_DATA_DIR, MODELS_DIR, IMAGE_SIZE, LEARNING_RATE

# Optimized hyperparameters for best results
BATCH_SIZE = 8  # Small batches = better generalization
EPOCHS_STAGE1 = 15  # Adequate for head training
EPOCHS_STAGE2 = 30  # Sufficient for fine-tuning
DROPOUT_RATE = 0.5  # Strong regularization
UNFREEZE_LAYERS = 100  # Deep fine-tuning


def clahe_numpy_uint8(img):
    """Apply CLAHE preprocessing for MRI contrast enhancement"""
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    cl3 = np.stack([cl, cl, cl], axis=-1)
    return (cl3.astype('float32') / 255.0)


def create_stratified_split(data_dir, val_split=0.15):
    """
    Create stratified train/val split ensuring balanced classes
    """
    filepaths = []
    labels = []

    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    filepaths.append(str(img_path))
                    labels.append(class_name)

    # Stratified split to maintain class balance
    train_files, val_files, train_labels, val_labels = train_test_split(
        filepaths, labels, test_size=val_split, stratify=labels, random_state=42
    )

    train_df = pd.DataFrame({'filepath': train_files, 'label': train_labels})
    val_df = pd.DataFrame({'filepath': val_files, 'label': val_labels})

    print(f"\n📊 Stratified Split:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")

    return train_df, val_df


def create_generators(train_df, val_df):
    """
    Create generators with CLAHE + strong augmentation
    """

    def preprocessing_with_clahe(img):
        """Apply CLAHE then EfficientNet preprocessing"""
        img_uint8 = np.clip(img, 0, 255).astype('uint8')
        # Apply CLAHE for better MRI contrast
        img_clahe = clahe_numpy_uint8(img_uint8)
        # Apply EfficientNet preprocessing
        return efficientnet_preprocess(img_clahe * 255.0)  # Scale back for preprocess_input

    # Strong augmentation to prevent overfitting
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_with_clahe,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation only uses CLAHE + preprocessing (no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_with_clahe
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen


def calculate_class_weights(train_gen):
    """Calculate balanced class weights"""
    from sklearn.utils.class_weight import compute_class_weight

    class_counts = {}
    for class_name, class_idx in train_gen.class_indices.items():
        class_counts[class_name] = np.sum(train_gen.classes == class_idx)

    total = sum(class_counts.values())

    cw = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(cw))

    print("\n📊 Class Distribution:")
    for name, count in sorted(class_counts.items()):
        pct = count / total * 100
        print(f"   {name:12s}: {count:4d} samples ({pct:5.1f}%)")

    print("\n⚖️  Balanced Class Weights:")
    for idx, weight in class_weights.items():
        class_name = list(train_gen.class_indices.keys())[idx]
        print(f"   {class_name:12s}: {weight:.3f}")

    return class_weights


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#2E86DE')
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#EE5A6F')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, color='#2E86DE')
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#EE5A6F')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, test_gen):
    """Comprehensive evaluation with per-class metrics"""
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    test_gen.reset()
    print("\nGenerating predictions...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())

    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT:")
    print("-" * 70)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX:")
    print("-" * 70)
    cm = confusion_matrix(y_true, y_pred)

    # Pretty print confusion matrix
    header = "True\\Pred".ljust(12) + "".join([f"{name[:10]:>12s}" for name in class_names])
    print(header)
    print("-" * 70)
    for i, name in enumerate(class_names):
        row = f"{name[:10]:12s}" + "".join([f"{cm[i][j]:12d}" for j in range(len(class_names))])
        print(row)

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("\n" + "-" * 70)
    print("PER-CLASS ACCURACY:")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        acc = class_accuracy[i] * 100
        bar = "█" * int(acc / 5)
        print(f"   {class_name:12s}: {acc:6.2f}% {bar}")

    return y_true, y_pred, cm, class_accuracy


def get_callbacks(model_path, patience):
    """Create training callbacks with proper monitoring"""
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(3, patience // 3),
            verbose=1,
            min_lr=1e-8
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]


def train_model(train_dir: Path, test_dir: Path, model_save_path: Path):
    """
    Complete training pipeline with all optimizations:
    - CLAHE preprocessing for MRI contrast
    - Stratified split for balanced validation
    - Strong augmentation to prevent overfitting
    - Two-stage training (head -> fine-tune)
    - Class weights for imbalance handling
    """

    print("=" * 70)
    print("🧠 BRAIN TUMOR CLASSIFICATION - OPTIMIZED TRAINING")
    print("=" * 70)
    print("\n✨ Features enabled:")
    print("   ✓ CLAHE preprocessing (MRI contrast enhancement)")
    print("   ✓ Stratified train/val split")
    print("   ✓ Strong augmentation (rotation, zoom, brightness, etc.)")
    print("   ✓ Two-stage training (head → fine-tune)")
    print("   ✓ High dropout (0.5) + L2 regularization")
    print("   ✓ Balanced class weights")
    print("   ✓ Small batch size (8) for better generalization")

    # Set MLflow experiment
    mlflow.set_experiment("brain-tumor-optimized")

    with mlflow.start_run():

        # Log all parameters
        mlflow.log_params({
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs_stage1": EPOCHS_STAGE1,
            "epochs_stage2": EPOCHS_STAGE2,
            "dropout_rate": DROPOUT_RATE,
            "unfreeze_layers": UNFREEZE_LAYERS,
            "preprocessing": "CLAHE + EfficientNet",
            "augmentation": "strong",
            "model": "EfficientNetB0"
        })

        # Create stratified split
        print("\n" + "=" * 70)
        print("STEP 1: Creating stratified train/val split...")
        print("=" * 70)
        train_df, val_df = create_stratified_split(train_dir, val_split=0.15)

        # Create generators
        print("\n" + "=" * 70)
        print("STEP 2: Creating data generators with CLAHE...")
        print("=" * 70)
        train_gen, val_gen = create_generators(train_df, val_df)

        # Create test generator
        def preprocessing_with_clahe(img):
            img_uint8 = np.clip(img, 0, 255).astype('uint8')
            img_clahe = clahe_numpy_uint8(img_uint8)
            return efficientnet_preprocess(img_clahe * 255.0)

        test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_with_clahe)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        num_classes = len(train_gen.class_indices)
        class_names = list(train_gen.class_indices.keys())

        print(f"\n✅ Data loaded:")
        print(f"   Classes: {class_names}")
        print(f"   Training:   {len(train_df):4d} samples")
        print(f"   Validation: {len(val_df):4d} samples")
        print(f"   Testing:    {test_gen.samples:4d} samples")

        # Calculate class weights
        class_weights = calculate_class_weights(train_gen)

        # Create model
        print("\n" + "=" * 70)
        print("STEP 3: Building EfficientNetB0 model...")
        print("=" * 70)
        model, base_model = create_efficientnet_model(
            input_shape=(*IMAGE_SIZE, 3),
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE
        )

        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

        print(f"✅ Model created:")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # ==================== STAGE 1: Train Head ====================
        print("\n" + "=" * 70)
        print("STAGE 1: Training classification head (base frozen)")
        print("=" * 70)

        model = compile_model(model, learning_rate=LEARNING_RATE, stage='head')
        callbacks_stage1 = get_callbacks(model_save_path / "stage1_best.keras", patience=10)

        print(f"\nTraining for up to {EPOCHS_STAGE1} epochs...")
        print(f"Learning rate: {LEARNING_RATE}")

        history_stage1 = model.fit(
            train_gen,
            epochs=EPOCHS_STAGE1,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks_stage1,
            verbose=1
        )

        stage1_acc = history_stage1.history['val_accuracy'][-1]
        print(f"\n✅ Stage 1 complete! Best val accuracy: {stage1_acc * 100:.2f}%")

        # ==================== STAGE 2: Fine-tune ====================
        print("\n" + "=" * 70)
        print("STAGE 2: Fine-tuning deep layers")
        print("=" * 70)

        model = unfreeze_model(model, base_model, num_layers_to_unfreeze=UNFREEZE_LAYERS)

        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"   Trainable parameters: {trainable_params:,}")

        # Much lower learning rate for fine-tuning
        fine_tune_lr = LEARNING_RATE / 20
        model = compile_model(model, learning_rate=fine_tune_lr, stage='finetune')

        callbacks_stage2 = get_callbacks(model_save_path / "stage2_best.keras", patience=15)

        print(f"\nTraining for up to {EPOCHS_STAGE2} epochs...")
        print(f"Learning rate: {fine_tune_lr}")

        history_stage2 = model.fit(
            train_gen,
            epochs=EPOCHS_STAGE2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks_stage2,
            verbose=1
        )

        # Combine histories
        history_combined = {
            'accuracy': history_stage1.history['accuracy'] + history_stage2.history['accuracy'],
            'val_accuracy': history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
            'loss': history_stage1.history['loss'] + history_stage2.history['loss'],
            'val_loss': history_stage1.history['val_loss'] + history_stage2.history['val_loss']
        }

        class HistoryWrapper:
            def __init__(self, hist_dict):
                self.history = hist_dict

        # Save training plot
        plot_path = model_save_path / "training_history.png"
        plot_training_history(HistoryWrapper(history_combined), plot_path)
        mlflow.log_artifact(plot_path)
        print(f"\n📊 Training curves saved: {plot_path}")

        # Final evaluation
        y_true, y_pred, cm, class_acc = evaluate_model(model, test_gen)

        test_gen.reset()
        test_metrics = model.evaluate(test_gen, verbose=0, return_dict=True)

        # Calculate metrics
        final_train_acc = history_stage2.history['accuracy'][-1]
        final_val_acc = history_stage2.history['val_accuracy'][-1]
        test_acc = test_metrics['accuracy']
        overfitting_gap = final_train_acc - final_val_acc

        # Log to MLflow
        mlflow.log_metrics({
            "final_train_accuracy": final_train_acc,
            "final_val_accuracy": final_val_acc,
            "test_accuracy": test_acc,
            "overfitting_gap": overfitting_gap,
            "min_class_accuracy": float(class_acc.min()),
            "max_class_accuracy": float(class_acc.max())
        })

        # Save model
        final_model_path = model_save_path / "brain_tumor_model.h5"
        model.save(final_model_path)
        mlflow.keras.log_model(model, "model")

        # Save class names
        import json
        class_names_path = model_save_path / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)

        # Final report
        print("\n" + "=" * 70)
        print("🎯 FINAL RESULTS")
        print("=" * 70)
        print(f"\n📈 Accuracy Metrics:")
        print(f"   Training:   {final_train_acc * 100:6.2f}%")
        print(f"   Validation: {final_val_acc * 100:6.2f}%")
        print(f"   Test:       {test_acc * 100:6.2f}%")
        print(f"\n📊 Model Quality:")
        print(f"   Overfitting gap:  {overfitting_gap * 100:6.2f}% {'✅' if overfitting_gap < 0.10 else '⚠️'}")
        print(f"   Min class acc:    {class_acc.min() * 100:6.2f}%")
        print(f"   Max class acc:    {class_acc.max() * 100:6.2f}%")
        print(f"\n💾 Saved Files:")
        print(f"   Model: {final_model_path}")
        print(f"   Class names: {class_names_path}")
        print(f"   Training plot: {plot_path}")
        print("=" * 70)

        # Performance assessment
        if test_acc > 0.85 and overfitting_gap < 0.10 and class_acc.min() > 0.70:
            print("\n🎉 EXCELLENT! Model is production-ready!")
            print("   ✓ High test accuracy (>85%)")
            print("   ✓ Low overfitting (<10%)")
            print("   ✓ All classes perform well (>70%)")
        elif test_acc > 0.75:
            print("\n👍 GOOD performance!")
            if overfitting_gap > 0.10:
                print("   ⚠️  Some overfitting detected")
            if class_acc.min() < 0.70:
                print(f"   ⚠️  Weakest class needs improvement: {class_names[class_acc.argmin()]}")
        else:
            print("\n⚠️  Performance needs improvement")
            print("   💡 Suggestions:")
            print("      - Check data quality")
            print("      - Ensure enough samples per class")
            print("      - Consider collecting more data")

        return model, history_combined, class_names


if __name__ == "__main__":

    train_data_path = RAW_DATA_DIR / "Training"
    test_data_path = RAW_DATA_DIR / "Testing"

    # Validate paths
    if not train_data_path.exists():
        print(f"❌ ERROR: Training data not found at {train_data_path}")
        print(f"\n📁 Expected structure:")
        print(f"   {train_data_path}/")
        print(f"   ├── glioma/")
        print(f"   ├── meningioma/")
        print(f"   ├── notumor/")
        print(f"   └── pituitary/")
        sys.exit(1)

    if not test_data_path.exists():
        print(f"❌ ERROR: Testing data not found at {test_data_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🚀 Starting optimized training with all enhancements")
    print("=" * 70)

    # Train with all optimizations
    model, history, class_names = train_model(
        train_dir=train_data_path,
        test_dir=test_data_path,
        model_save_path=MODELS_DIR
    )

    print("\n✨ Training complete!")
    print("\n📌 Next steps:")
    print("   1. View experiments: mlflow ui")
    print("   2. Start API: uvicorn src.api.main:app --reload")
    print("   3. Test endpoint: http://localhost:8000/docs")