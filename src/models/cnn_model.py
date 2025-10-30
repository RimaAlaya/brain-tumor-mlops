import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_data_augmentation():
    """
    Create data augmentation layer to reduce overfitting
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")


def create_cnn_model(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.5):
    """
    Create CNN model for brain tumor classification with regularization
    THIS IS A FALLBACK - Use create_efficientnet_model for better results

    Args:
        input_shape: Shape of input images
        num_classes: Number of tumor classes
        dropout_rate: Dropout rate for regularization

    Returns:
        Keras model
    """

    # Weight decay for L2 regularization
    weight_decay = 1e-4

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Data augmentation (only active during training)
        create_data_augmentation(),

        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Fourth conv block
        layers.Conv2D(256, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='brain_tumor_cnn')

    return model


def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.5):
    """
    Create EfficientNetB0 model for brain tumor classification
    THIS IS THE RECOMMENDED MODEL - Much better accuracy than basic CNN

    Args:
        input_shape: Shape of input images (must be 224x224 for EfficientNet)
        num_classes: Number of tumor classes
        dropout_rate: Dropout rate for regularization (0.5 recommended)

    Returns:
        Tuple of (model, base_model) for two-stage training
    """

    # Load EfficientNetB0 with ImageNet weights
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze base initially for stage 1 training
    for layer in base.layers:
        layer.trainable = False

    # Build classification head with DOUBLE dropout layers
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(dropout_rate * 0.7)(x)  # Second dropout layer
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out, name='brain_tumor_efficientnet')

    return model, base


def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.5):
    """
    Wrapper for create_efficientnet_model for backward compatibility
    """
    model, base = create_efficientnet_model(input_shape, num_classes, dropout_rate)
    return model


def unfreeze_model(model, base_model, num_layers_to_unfreeze=50):
    """
    Unfreeze last N layers of base model for fine-tuning
    Call this after initial training with frozen base

    Args:
        model: Full model
        base_model: Base model (EfficientNet)
        num_layers_to_unfreeze: Number of last layers to unfreeze

    Returns:
        Model with unfrozen layers
    """
    # Unfreeze last N layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    print(f"✅ Unfroze last {num_layers_to_unfreeze} layers for fine-tuning")

    return model


def compile_model(model, learning_rate=0.001, stage='head'):
    """
    Compile the model with optimizer and metrics

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        stage: 'head' for initial training, 'finetune' for fine-tuning

    Returns:
        Compiled model
    """

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model compiled with lr={learning_rate:.2e} for {stage} training")

    return model


def get_callbacks(model_save_path, patience=7, monitor='val_accuracy'):
    """
    Get training callbacks

    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping
        monitor: Metric to monitor

    Returns:
        List of callbacks
    """

    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),

        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks