# 🧠 Brain Tumor Classification MLOps Pipeline

[![CI Pipeline](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/ci.yml)
[![Docker Build](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/docker.yml/badge.svg)](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/docker.yml)
[![Code Quality](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/lint.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready end-to-end MLOps pipeline for automated brain tumor classification from MRI scans using deep learning**

**🎯 Achieved 98.5% test accuracy** with comprehensive MLOps infrastructure including automated training, model versioning, containerized deployment, and interactive demo.

---

## 📊 Project Overview

A complete MLOps implementation demonstrating industry best practices for deploying deep learning models in production. This project classifies brain tumors from MRI scans into 4 categories with high accuracy while showcasing:

- ✅ **Production ML Pipeline** - Automated training with experiment tracking
- ✅ **Model Registry** - Version management with automated promotion
- ✅ **REST API** - FastAPI with 8 endpoints and auto-documentation
- ✅ **Interactive Demo** - Gradio web interface with real-time predictions
- ✅ **Docker Deployment** - Multi-container architecture with health checks
- ✅ **CI/CD Pipeline** - Automated testing, linting, and Docker builds
- ✅ **Comprehensive Testing** - 85%+ code coverage with pytest

### 🎯 Classification Categories

| Class | Description |
|-------|-------------|
| **Glioma** | Tumor occurring in the brain and spinal cord |
| **Meningioma** | Tumor arising from the meninges |
| **Pituitary** | Tumor in the pituitary gland |
| **No Tumor** | No tumor detected in MRI scan |

---

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 99.96% |
| **Validation Accuracy** | 96.94% |
| **Test Accuracy** | **98.47%** ⭐ |
| **Inference Time** | ~50ms per image |
| **Model Size** | 16.2 MB (.keras format) |
| **Training Time** | 65.9 minutes (12 epochs) |

**Architecture:** EfficientNetB0 with transfer learning  
**Framework:** TensorFlow 2.18.0  
**Input Size:** 224×224×3 RGB images  
**Total Parameters:** 4,049,564 (all trainable)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- 4GB+ RAM

### Option 1: Interactive Demo (Fastest)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/brain-tumor-mlops.git
cd brain-tumor-mlops

# Install dependencies
pip install -r requirements.txt

# Launch Gradio demo
python run_demo.py --share

# Access at: http://localhost:7860
# Or use the public Gradio link for sharing
```

### Option 2: REST API

```bash
# Using Docker (Recommended)
docker-compose up -d api

# Or run locally
uvicorn src.api.main:app --reload

# Access at: http://localhost:8000/docs
```

### Option 3: Train Your Own Model

```bash
# Prepare your data in data/raw/Training and data/raw/Testing

# Train model with MLflow tracking
python src/training/train.py

# View experiments
mlflow ui
# Open: http://localhost:5000
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface Layer                   │
├──────────────────────┬──────────────────────────────────┤
│   Gradio Web Demo    │      FastAPI REST API            │
│   (Port 7860)        │      (Port 8000)                 │
└──────────┬───────────┴───────────┬──────────────────────┘
           │                       │
           ▼                       ▼
    ┌──────────────────────────────────────┐
    │     EfficientNetB0 Model             │
    │     (98.5% Accuracy)                 │
    └──────────────┬───────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   MLflow Registry   │
         │   Model Versioning  │
         └─────────────────────┘
```

### Project Structure

```
brain-tumor-mlops/
├── src/
│   ├── api/                  # FastAPI application
│   │   ├── main.py          # API endpoints (8 routes)
│   │   └── schemas.py       # Pydantic models
│   ├── models/              # Model architectures
│   │   └── cnn_model.py    # EfficientNetB0 builder
│   ├── training/            # Training pipeline
│   │   └── train.py        # Enhanced training script
│   ├── demo/                # Interactive demo
│   │   └── gradio_app.py   # Gradio interface
│   ├── mlflow_utils/        # MLflow utilities
│   │   ├── registry.py     # Model registry
│   │   ├── tracking.py     # Experiment tracking
│   │   └── comparison.py   # Experiment comparison
│   └── config.py           # Configuration
├── tests/                   # Comprehensive test suite
│   ├── test_api.py         # API endpoint tests
│   ├── test_model.py       # Model tests
│   └── test_data.py        # Data pipeline tests
├── docker/                  # Docker configurations
│   ├── Dockerfile.train    # Training container
│   └── Dockerfile.serve    # Serving container
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml             # Main CI pipeline
│   ├── docker.yml         # Docker build & push
│   └── lint.yml           # Code quality checks
├── models/                 # Saved models
├── data/                   # Dataset
├── scripts/               # Utility scripts
├── docker-compose.yml     # Service orchestration
└── requirements.txt       # Python dependencies
```

---

## 📡 API Endpoints

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health with statistics |
| `GET` | `/model/info` | Model architecture details |
| `GET` | `/classes` | Available classification classes |
| `GET` | `/stats` | API usage statistics |
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | Batch prediction (max 10 images) |
| `GET` | `/docs` | Interactive API documentation |

### Example Usage

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
files = {"file": open("mri_scan.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['all_probabilities']}")

# Response:
# {
#   "predicted_class": "glioma",
#   "confidence": 0.9847,
#   "all_probabilities": {
#     "glioma": 0.9847,
#     "meningioma": 0.0098,
#     "notumor": 0.0032,
#     "pituitary": 0.0023
#   },
#   "inference_time_seconds": 0.0523,
#   "timestamp": "2025-11-09T12:30:45"
# }
```

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start API service
docker-compose up -d api

# Start with MLflow UI
docker-compose --profile mlflow up -d

# Run training in container
docker-compose --profile training run --rm train

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

### Manual Docker Commands

```bash
# Build serving image
docker build -f docker/Dockerfile.serve -t brain-tumor-api .

# Run API container
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name brain-tumor-api \
  brain-tumor-api

# Check health
curl http://localhost:8000/health
```

---

## 🧪 Testing

### Run All Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```


---

## 📊 MLflow Experiment Tracking

### Features

- ✅ **Automatic experiment logging** - Parameters, metrics, artifacts
- ✅ **Model registry** - Version management with stages (Dev/Staging/Production)
- ✅ **Automated promotion** - Best models auto-promoted based on accuracy threshold
- ✅ **Experiment comparison** - Visual comparison of multiple training runs
- ✅ **Rich metadata** - Git commits, system info, dataset versions

### View Experiments

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser
# http://localhost:5000
```

### Model Registry

```python
from src.mlflow_utils import ModelRegistry

registry = ModelRegistry()

# Get production model
prod_model = registry.get_production_model("brain_tumor_classifier")

# Get best model by accuracy
best_model = registry.get_best_model("brain_tumor_classifier", metric="val_accuracy")

# Promote model to production
registry.promote_model(
    model_name="brain_tumor_classifier",
    version=3,
    stage="Production"
)
```

---

## 🎨 Interactive Demo

### Gradio Web Interface

Launch the interactive demo to test the model with your own MRI images:

```bash
# Start demo
python run_demo.py

# With public sharing link
python run_demo.py --share

# Custom port
python run_demo.py --port 8080
```

**Demo Features:**
- 📤 Drag-and-drop image upload
- ⚡ Real-time predictions (<100ms)
- 📊 Confidence score visualization
- 📝 Detailed class descriptions
- 🔗 Shareable public links

**Live Demo:** [Try it here](https://7a125e31dd06ddece7.gradio.live) *(link expires in 72 hours)*

---

## 🔄 CI/CD Pipeline

### Automated Workflows

Every push triggers:

1. **Code Quality Checks** (lint.yml)
   - flake8 linting
   - black formatting check
   - isort import sorting
   - mypy type checking

2. **Testing Pipeline** (ci.yml)
   - Unit tests with pytest
   - Coverage reporting (85%+)
   - Security scanning with Bandit
   - Docker image build test

3. **Docker Build & Push** (docker.yml)
   - Multi-stage image builds
   - Push to GitHub Container Registry
   - Trivy security scanning
   - Automatic versioning

### Status Badges

View current build status in the badges at the top of this README.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | TensorFlow 2.18.0, Keras |
| **Model Architecture** | EfficientNetB0 (Transfer Learning) |
| **API Framework** | FastAPI 0.109.0 |
| **Web Interface** | Gradio 4.15.0 |
| **Experiment Tracking** | MLflow 2.9.2 |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest, pytest-cov |
| **Code Quality** | black, flake8, isort, pylint |
| **Data Processing** | NumPy, pandas, OpenCV, Pillow |
| **Visualization** | Matplotlib, seaborn |

---

## 📈 Model Training Details

### Hyperparameters

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001
OPTIMIZER = "Adam"
LOSS = "categorical_crossentropy"
VALIDATION_SPLIT = 0.10
```

### Data Augmentation

- EfficientNet standard preprocessing
- Train/Test split: 90/10
- Total images: 3,000+ MRI scans
- Classes balanced during training

### Training Configuration

- **Full model fine-tuning** (all layers trainable)
- **Callbacks:** ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
- **Metrics:** Accuracy, Loss (tracked per epoch)
- **Hardware:** GPU-accelerated (when available)

### Reproducibility

All experiments tracked with:
- Git commit hash
- Python/TensorFlow versions
- System specifications
- Random seeds set
- Dataset versions

---

## 🔒 Model Versioning

### MLflow Model Registry

Models are automatically versioned and managed through stages:

```
brain_tumor_classifier
├── v1 (Archived) - 91.2% accuracy
├── v2 (Development) - 94.5% accuracy
├── v3 (Staging) - 96.9% accuracy
└── v4 (Production) ⭐ - 98.5% accuracy
```

### Automated Promotion

Models are automatically promoted to Production when:
- Validation accuracy > 90%
- Test accuracy > 85%
- All tests pass
- Better than current production model

---

## 📖 Documentation

- [API Setup Guide](API_SETUP_GUIDE.md) - FastAPI configuration
- [Docker Guide](DOCKER_GUIDE.md) - Container deployment
- [MLflow Guide](MLFLOW_ENHANCED_GUIDE.md) - Experiment tracking
- [CI/CD Guide](CICD_SETUP_GUIDE.md) - Pipeline configuration
- [Gradio Demo Guide](GRADIO_DEMO_GUIDE.md) - Interactive interface

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass (`pytest tests/`)
- Code is formatted (`black src/ tests/`)
- Imports are sorted (`isort src/ tests/`)
- Linting passes (`flake8 src/`)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rima Alaya**
- GitHub: [@RimaAlaya](https://github.com/RimaAlaya)
- LinkedIn: [Rima Alaya](www.linkedin.com/in/rima-alaya)
- Email: rimaalaya76@gmail.com


---

## 🙏 Acknowledgments

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Base Model: EfficientNet by Google Research
- Framework: TensorFlow/Keras team
- MLOps Tools: MLflow, FastAPI, Gradio communities

---

## 📊 Project Statistics

- **Lines of Code:** 3,500+ (Python)
- **Test Coverage:** 87%
- **API Endpoints:** 8
- **Docker Images:** 3 (train, serve, mlflow)
- **CI/CD Workflows:** 3
- **Model Versions:** 4+
- **Total Commits:** 50+




## ⭐ Star History

If you find this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=RimaAlaya/brain-tumor-mlops&type=Date)](https://star-history.com/#YOUR_USERNAME/brain-tumor-mlops&Date)

---

<div align="center">

**Built with ❤️ using TensorFlow, FastAPI, and MLflow**

[🏠 Home](https://github.com/RimaAlaya/brain-tumor-mlops) • 
[📖 Docs](https://github.com/RimaAlaya/brain-tumor-mlops/tree/main/docs) • 


Made with 🧠 for advancing medical AI

</div>