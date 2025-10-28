import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


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


def create_cnn_model(input_shape=(150, 150, 3), num_classes=4, dropout_rate=0.5):
    """
    Create CNN model for brain tumor classification with regularization

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


def create_transfer_learning_model(input_shape=(150, 150, 3), num_classes=4, dropout_rate=0.5):
    """
    Create transfer learning model using EfficientNetB0
    Often performs better with limited data

    Args:
        input_shape: Shape of input images
        num_classes: Number of tumor classes
        dropout_rate: Dropout rate for regularization

    Returns:
        Keras model
    """

    # Load pretrained EfficientNetB0
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze base model initially
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=input_shape)

    # Data augmentation
    x = create_data_augmentation()(inputs)

    # Pretrained base
    x = base_model(x, training=False)

    # Custom head
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='brain_tumor_efficientnet')

    return model


def unfreeze_model(model, unfreeze_from_layer=100):
    """
    Unfreeze layers for fine-tuning
    Call this after initial training with frozen base

    Args:
        model: Model to unfreeze
        unfreeze_from_layer: Layer index to start unfreezing from
    """
    base_model = model.layers[1]  # Assuming augmentation is first layer
    base_model.trainable = True

    # Freeze early layers, unfreeze later ones
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and metrics

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled model
    """

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Simplified metrics to avoid shape issues
    )

    return model


def get_callbacks(model_save_path, patience=7):
    """
    Get training callbacks

    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping

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
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),

        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks