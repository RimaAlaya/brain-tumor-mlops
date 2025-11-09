import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn_model import build_effnet, get_callbacks


def test_model_creation():
    """Test model can be created"""
    model = build_effnet(image_size=(224, 224), num_classes=4, dropout_rate=0.5, learning_rate=1e-3)
    assert model is not None
    assert len(model.layers) > 0


def test_model_compilation():
    """Test model is compiled properly"""
    model = build_effnet(image_size=(224, 224), num_classes=4)
    # Model is already compiled in build_effnet
    assert model.optimizer is not None
    assert model.loss is not None


def test_callbacks():
    """Test callbacks can be created"""
    from pathlib import Path

    test_path = Path("test_model.keras")

    callbacks = get_callbacks(test_path)
    assert callbacks is not None
    assert len(callbacks) == 3  # checkpoint, reduce_lr, early_stopping

    # Cleanup
    if test_path.exists():
        test_path.unlink()


def test_model_output_shape():
    """Test model output shape is correct"""
    model = build_effnet(image_size=(224, 224), num_classes=4)
    # Output should be (None, 4) for 4 classes
    assert model.output_shape == (None, 4)


def test_model_input_shape():
    """Test model input shape is correct"""
    model = build_effnet(image_size=(224, 224), num_classes=4)
    # Input should be (None, 224, 224, 3)
    assert model.input_shape == (None, 224, 224, 3)
