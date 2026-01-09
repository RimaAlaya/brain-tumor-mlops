<div align="center">

# 🧠 Brain Tumor Classification MLOps Pipeline

[![CI Pipeline](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/ci.yml)
[![Docker Build](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/docker.yml/badge.svg)](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/docker.yml)
[![Code Quality](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/lint.yml/badge.svg)](https://github.com/RimaAlaya/brain-tumor-mlops/actions/workflows/lint.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Enterprise-Grade Deep Learning System for Automated Brain Tumor Classification

**Production-ready MLOps pipeline achieving 98.5% accuracy** with comprehensive CI/CD, automated training, model versioning, containerized deployment, and interactive web interface.

[🚀 Live Demo](https://brain-tumor-mlops.onrender.com) • [📖 API Docs](https://brain-tumor-mlops.onrender.com/docs) • [🐳 Docker Hub](https://hub.docker.com/r/rimaalaya/brain-tumor-api)

---

![Brain Tumor Classifier Demo](https://raw.githubusercontent.com/RimaAlaya/brain-tumor-mlops/images/image.png)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [API Endpoints](#-api-endpoints)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring](#-monitoring)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)

---

## 🎯 Overview

A **production-grade end-to-end MLOps pipeline** for automated brain tumor classification from MRI scans. This project demonstrates industry best practices for deploying deep learning models in production environments.

### 🏥 Classification Categories

| Type | Description | Clinical Significance |
|------|-------------|----------------------|
| 🔬 **Glioma** | Brain/spinal cord tumor | Immediate attention needed |
| 🧬 **Meningioma** | Meninges tumor | Often benign |
| ⚡ **Pituitary** | Pituitary gland tumor | Hormone effects |
| ✅ **No Tumor** | Healthy brain tissue | No abnormality |

> ⚠️ **Medical Disclaimer**: AI demonstration for educational purposes only. Always consult qualified medical professionals.

---

## 🎮 Live Demo

### 🌐 Interactive Web Application

<div align="center">

### 🔗 [**Try Live Demo →**](https://brain-tumor-mlops.onrender.com)

**Features:**
- 📤 Drag-and-drop image upload
- ⚡ Real-time predictions (<100ms)
- 📊 Visual confidence scores
- 🎨 Modern, responsive UI

</div>

### 🔌 API Access

```python
import requests

url = "https://brain-tumor-mlops.onrender.com/predict"
files = {"file": open("mri_scan.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())

# Output:
# {
#   "predicted_class": "glioma",
#   "confidence": 0.9847,
#   "inference_time_seconds": 0.052
# }
```

**📖 Interactive Docs:** [https://brain-tumor-mlops.onrender.com/docs](https://brain-tumor-mlops.onrender.com/docs)

---

## 🌟 Key Features

### 🔧 MLOps Infrastructure
- ✅ Automated training pipeline with MLflow experiment tracking
- ✅ Model registry & versioning with automated promotion
- ✅ Docker containerization for reproducibility
- ✅ CI/CD pipelines with GitHub Actions
- ✅ Production monitoring with Prometheus & Grafana

### 🚀 Deployment & APIs
- ✅ FastAPI REST API with 8 endpoints + auto-generated docs
- ✅ Interactive Gradio demo deployed on Render
- ✅ Batch prediction support (up to 10 images)
- ✅ Health checks with detailed metrics
- ✅ CORS support for web integration

### 🧪 Quality Assurance
- ✅ 87% test coverage with pytest
- ✅ Code quality checks (black, flake8, isort, mypy)
- ✅ Security scanning (Bandit, Trivy)
- ✅ Dockerfile linting with hadolint

---

## 🏆 Model Performance

<div align="center">

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **98.47%** ⭐ |
| **Validation Accuracy** | 96.94% |
| **Training Accuracy** | 99.96% |
| **Inference Time** | ~50ms |
| **Model Size** | 16.2 MB |
| **Training Time** | 65.9 min (12 epochs) |

</div>

**Architecture:** EfficientNetB0 (Transfer Learning)  
**Framework:** TensorFlow/Keras 2.18.0  
**Parameters:** 4,049,564 (all trainable)

---

## 🚀 Quick Start

### 1️⃣ Interactive Demo (Fastest)

```bash
git clone https://github.com/RimaAlaya/brain-tumor-mlops.git
cd brain-tumor-mlops
pip install -r requirements.txt
python run_demo.py

# 🎉 Access at: http://localhost:7860
```

### 2️⃣ REST API (Production)

```bash
# Using Docker (Recommended)
docker-compose up -d api

# 📖 Docs: http://localhost:8000/docs
# 🔍 Health: http://localhost:8000/health
```

### 3️⃣ Train Custom Model

```bash
# Prepare data in data/raw/Training and data/raw/Testing
python src/training/train.py

# View experiments
mlflow ui  # http://localhost:5000
```

### 4️⃣ Full Stack with Monitoring

```bash
docker-compose up -d

# API: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│              User Interface Layer                        │
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

### MLOps Workflow

```
Data → Training → MLflow → Registry → Promotion → Deployment → Monitoring → Feedback
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Basic health check |
| `GET` | `/health` | Detailed health + stats |
| `GET` | `/model/info` | Model architecture |
| `GET` | `/classes` | Classification classes |
| `GET` | `/stats` | API usage statistics |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch prediction (max 10) |
| `GET` | `/docs` | Swagger documentation |
| `GET` | `/metrics` | Prometheus metrics |

### Usage Examples

**Single Prediction:**
```python
import requests

response = requests.post(
    "https://brain-tumor-mlops.onrender.com/predict",
    files={"file": open("mri.jpg", "rb")}
)
result = response.json()
print(f"{result['predicted_class']}: {result['confidence']:.2%}")
```

**Batch Prediction:**
```python
files = [("files", open(f"scan{i}.jpg", "rb")) for i in range(3)]
response = requests.post(
    "https://brain-tumor-mlops.onrender.com/predict/batch",
    files=files
)
```

---

## 🐳 Docker Deployment

### Quick Deploy

```bash
# Start API
docker-compose up -d api

# Full stack (API + Monitoring)
docker-compose up -d

# Training
docker-compose --profile training run --rm train
```

### Manual Docker

```bash
# Build
docker build -f docker/Dockerfile.serve -t brain-tumor-api .

# Run
docker run -d -p 8000:8000 brain-tumor-api

# From Docker Hub
docker pull rimaalaya/brain-tumor-api:latest
docker run -d -p 8000:8000 rimaalaya/brain-tumor-api:latest
```

---

## 🔄 CI/CD Pipeline

### Automated Workflows

Every push triggers:

1. **Code Quality** (lint.yml)
   - Black formatting, isort, flake8, mypy
   - Dockerfile linting with hadolint

2. **Testing** (ci.yml)
   - Unit tests with 87% coverage
   - Security scanning with Bandit
   - Docker build verification

3. **Docker Build** (docker.yml)
   - Multi-stage builds
   - Push to GitHub Container Registry
   - Trivy security scanning
   - Auto-versioning

**Deployment:** `main` branch → CI/CD → Docker Build → Render Deploy → Production

---

## 📊 Monitoring

### Prometheus Metrics

```python
brain_tumor_predictions_total          # Total predictions
brain_tumor_prediction_latency_seconds # Latency histogram
brain_tumor_prediction_confidence      # Confidence distribution
brain_tumor_predictions_by_class_total # Per-class counts
brain_tumor_model_loaded               # Model status
brain_tumor_errors_total               # Error tracking
```

### Grafana Dashboards

Professional monitoring with:
- 📊 Request rate and latency
- 🎯 Predictions by class
- 📈 P95 latency tracking
- ❌ Error rates
- 🤖 Model performance

**Access:** http://localhost:3000 (admin/admin)

---

## 🛠️ Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-0194E2?logo=mlflow)](https://mlflow.org/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana)](https://grafana.com/)

</div>

**Core:** TensorFlow/Keras, EfficientNetB0, FastAPI, Gradio  
**MLOps:** MLflow, Prometheus, Grafana, Docker, GitHub Actions  
**Testing:** pytest, black, flake8, isort, Bandit, Trivy

---

## 📂 Project Structure

```
brain-tumor-mlops/
├── src/
│   ├── api/              # FastAPI application
│   ├── models/           # Model architectures
│   ├── training/         # Training pipeline
│   ├── demo/            # Gradio interface
│   ├── mlflow_utils/    # MLflow utilities
│   └── monitoring/      # Prometheus metrics
├── tests/               # Test suite (87% coverage)
├── docker/              # Docker configurations
├── .github/workflows/   # CI/CD pipelines
├── grafana/            # Grafana dashboards
├── models/             # Saved models
├── data/               # Dataset
└── docker-compose.yml  # Service orchestration
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

**Ensure:**
```bash
black src/ tests/    # Format code
isort src/ tests/    # Sort imports
pytest tests/ -v     # Run tests
flake8 src/          # Check linting
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Rima Alaya**

[![GitHub](https://img.shields.io/badge/GitHub-RimaAlaya-181717?logo=github)](https://github.com/RimaAlaya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/rima-alaya)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail)](mailto:rimaalaya76@gmail.com)

---

## 🙏 Acknowledgments

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Model: EfficientNet by Google Research
- Framework: TensorFlow/Keras
- Tools: MLflow, FastAPI, Gradio

---

## ⭐ Star History

If this project helped you, please consider giving it a star!

[![Star History](https://api.star-history.com/svg?repos=RimaAlaya/brain-tumor-mlops&type=Date)](https://star-history.com/#RimaAlaya/brain-tumor-mlops&Date)

---

<div align="center">

**Built with ❤️ for advancing medical AI**

[🏠 Home](https://github.com/RimaAlaya/brain-tumor-mlops) • 
[📖 Docs](https://brain-tumor-mlops.onrender.com/docs) • 
[🚀 Demo](https://brain-tumor-mlops.onrender.com) • 
[📧 Contact](mailto:rimaalaya76@gmail.com)

Made with 🧠 for advancing medical AI • [MIT License](LICENSE)

</div>