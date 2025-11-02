"""
API Usage Examples for Brain Tumor Classification API

This script demonstrates how to use all endpoints of the API.
Make sure the API is running: uvicorn src.api.main:app --reload
"""

import requests
import json
from pathlib import Path
from typing import List

# API Configuration
API_BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def health_check():
    """Example: Check API health"""
    print_section("1. Health Check")

    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def detailed_health():
    """Example: Get detailed health information"""
    print_section("2. Detailed Health Check")

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def get_model_info():
    """Example: Get model information"""
    print_section("3. Get Model Information")

    response = requests.get(f"{API_BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Model: {data['model_name']}")
        print(f"Framework: {data['framework']}")
        print(f"Parameters: {data['total_parameters']:,}")
        print(f"Classes: {data['classes']}")
        print(f"Input Size: {data['image_size']}")
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def get_classes():
    """Example: Get available classes"""
    print_section("4. Get Available Classes")

    response = requests.get(f"{API_BASE_URL}/classes")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def predict_single_image(image_path: str):
    """Example: Predict single image"""
    print_section("5. Single Image Prediction")

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("Please provide a valid image path")
        return

    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/predict", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n🎯 Prediction Results:")
        print(f"   Predicted Class: {data['predicted_class']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print(f"   Inference Time: {data['inference_time_seconds']:.4f}s")
        print(f"\n📊 All Probabilities:")
        for cls, prob in data['all_probabilities'].items():
            print(f"   {cls:15} {prob:.2%}")
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def predict_batch_images(image_paths: List[str]):
    """Example: Predict multiple images"""
    print_section("6. Batch Image Prediction")

    # Validate images exist
    valid_paths = [p for p in image_paths if Path(p).exists()]

    if not valid_paths:
        print(f"❌ No valid images found")
        print("Please provide valid image paths")
        return

    print(f"Submitting {len(valid_paths)} images...")

    files = []
    for path in valid_paths:
        with open(path, 'rb') as f:
            files.append(('files', (Path(path).name, f.read(), 'image/jpeg')))

    response = requests.post(f"{API_BASE_URL}/predict/batch", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n📦 Batch Results:")
        print(f"   Total Images: {data['total_images']}")
        print(f"   Successful: {data['successful_predictions']}")
        print(f"   Total Time: {data['total_time_seconds']:.4f}s")

        print(f"\n🎯 Individual Predictions:")
        for pred in data['predictions']:
            if pred['error']:
                print(f"\n   ❌ {pred['filename']}: {pred['error']}")
            else:
                print(f"\n   ✅ {pred['filename']}")
                print(f"      Class: {pred['predicted_class']} ({pred['confidence']:.2%})")
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def get_statistics():
    """Example: Get API statistics"""
    print_section("7. Get API Statistics")

    response = requests.get(f"{API_BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all examples"""
    print("\n" + "🧠 Brain Tumor Classification API - Usage Examples ".center(60, "="))
    print(f"API URL: {API_BASE_URL}")

    try:
        # Basic checks
        health_check()
        detailed_health()
        get_model_info()
        get_classes()

        # Prediction examples
        # NOTE: Replace with actual image paths
        sample_image = "path/to/your/mri_image.jpg"
        sample_images = [
            "path/to/image1.jpg",
            "path/to/image2.jpg"
        ]

        print_section("📝 Note")
        print("To test predictions, please update the image paths in the script:")
        print(f"  - Single image: {sample_image}")
        print(f"  - Batch images: {sample_images}")

        # Uncomment these when you have actual images:
        # predict_single_image(sample_image)
        # predict_batch_images(sample_images)

        # Statistics
        get_statistics()

        print_section("✨ All Examples Completed")
        print("Check the API docs for more details: http://localhost:8000/docs")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running:")
        print("  uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    main()