from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model hyperparameters - optimized for brain tumor classification
IMAGE_SIZE = (224, 224)  # Standard size for pretrained models
BATCH_SIZE = 16  # Balance between speed and generalization
EPOCHS = 50  # With early stopping, will stop when converged
LEARNING_RATE = 0.0001  # Low learning rate for stable training