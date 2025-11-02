import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn_model import compile_model, create_cnn_model


def test_model_creation():
    """Test model can be created"""
    model = create_cnn_model(input_shape=(150, 150, 3), num_classes=4)
    assert model is not None
    assert len(model.layers) > 0


def test_model_compilation():
    """Test model can be compiled"""
    model = create_cnn_model()
    compiled_model = compile_model(model)
    assert compiled_model.optimizer is not None
