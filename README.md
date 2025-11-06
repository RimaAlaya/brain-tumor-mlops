# 🧠 Brain Tumor Classification MLOps Pipeline

> Production-ready deep learning pipeline for brain tumor MRI classification with end-to-end MLOps practices

# 🧠 Brain Tumor Classification MLOps Pipeline

[![CI Pipeline](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/ci.yml)
[![Docker Build](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/docker.yml/badge.svg)](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/docker.yml)
[![Code Quality](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/brain-tumor-mlops/actions/workflows/lint.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-ready deep learning pipeline for brain tumor MRI classification with end-to-end MLOps practices

## 🎯 Project Overview
...rest of your README...
## 🎯 Project Overview

A comprehensive MLOps implementation for classifying brain tumors from MRI scans into 4 categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

**Key Features:**
- ✅ Modular, production-ready codebase
- ✅ MLflow experiment tracking
- ✅ FastAPI REST API with automatic documentation
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions
- ✅ Comprehensive testing suite
- ✅ Interactive Gradio demo

## 🏗️ Architecture
```
brain-tumor-mlops/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training scripts
│   └── api/            # FastAPI application
├── tests/              # Unit tests
├── docker/             # Docker configurations
├── .github/workflows/  # CI/CD pipelines
└── models/             # Saved models
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-mlops.git
cd brain-tumor-mlops
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training
```bash
python src/training/train.py
```

View MLflow UI:
```bash
mlflow ui
```

### API Deployment

**Local:**
```bash
uvicorn src.api.main:app --reload
```

**Docker:**
```bash
docker-compose up
```

Access API docs at: `http://localhost:8000/docs`

### Testing
```bash
pytest tests/ -v --cov=src
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 95.2% |
| F1-Score | 0.94 |
| Inference Time | ~50ms |

## 🛠️ Tech Stack

- **ML Framework:** TensorFlow 2.15
- **Experiment Tracking:** MLflow
- **API:** FastAPI
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Testing:** pytest
- **UI:** Gradio

## 📝 API Usage
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("mri_scan.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)

---

⭐ Star this repo if you find it helpful!