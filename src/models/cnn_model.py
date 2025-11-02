from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_effnet(image_size, num_classes, dropout_rate=0.5, learning_rate=1e-3):
    """
    Build EfficientNetB0 model with full backbone training
    Matches your exact notebook approach
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3),
    )

    # Train whole backbone (like your notebook)
    for layer in base.layers:
        layer.trainable = True

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks(model_save_path):
    """Get training callbacks"""
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    checkpoint = ModelCheckpoint(
        str(model_save_path), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.3, patience=2, min_delta=0.001, verbose=1
    )
    early = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
    )

    return [checkpoint, reduce_lr, early]
