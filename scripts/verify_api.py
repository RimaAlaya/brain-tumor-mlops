#!/usr/bin/env python3
"""
API Verification Script

Run this script to verify that the enhanced API is working correctly.
Make sure the API is running first: uvicorn src.api.main:app --reload
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"


def check_mark(passed: bool) -> str:
    """Return check or cross mark"""
    return "✅" if passed else "❌"


def test_endpoint(name: str, method: str, endpoint: str, expected_status: int = 200) -> bool:
    """Test a single endpoint"""
    try:
        if method.upper() == "GET":
            response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        else:
            response = requests.post(f"{API_URL}{endpoint}", timeout=5)

        passed = response.status_code == expected_status
        status = check_mark(passed)
        print(f"{status} {name:35} Status: {response.status_code}")
        return passed
    except requests.exceptions.ConnectionError:
        print(f"❌ {name:35} ERROR: Connection refused")
        return False
    except Exception as e:
        print(f"❌ {name:35} ERROR: {str(e)}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print(" 🧠 Brain Tumor API - Verification Tests ".center(60))
    print("=" * 60 + "\n")

    print(f"Testing API at: {API_URL}\n")

    results = []

    # Test all endpoints
    print("📋 Endpoint Tests:")
    print("-" * 60)

    results.append(test_endpoint("Health Check (/)", "GET", "/"))
    results.append(test_endpoint("Detailed Health (/health)", "GET", "/health"))
    results.append(test_endpoint("Model Info (/model/info)", "GET", "/model/info"))
    results.append(test_endpoint("Classes (/classes)", "GET", "/classes"))
    results.append(test_endpoint("Statistics (/stats)", "GET", "/stats"))

    print("\n📚 Documentation Tests:")
    print("-" * 60)

    results.append(test_endpoint("Swagger UI (/docs)", "GET", "/docs"))
    results.append(test_endpoint("ReDoc (/redoc)", "GET", "/redoc"))
    results.append(test_endpoint("OpenAPI Schema (/openapi.json)", "GET", "/openapi.json"))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100

    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total}) - {percentage:.0f}%")
        print("\n✨ Your API is working perfectly!")
        print(f"\n📖 View API docs: {API_URL}/docs")
        print(f"📘 View ReDoc: {API_URL}/redoc")
    else:
        print(f"⚠️  SOME TESTS FAILED: ({passed}/{total}) - {percentage:.0f}%")
        print("\n🔍 Check the errors above and:")
        print("   1. Make sure the API is running")
        print("   2. Make sure the model is trained")
        print("   3. Check the logs for errors")

    print("=" * 60 + "\n")

    # Additional info
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("📊 API Statistics:")
            print("-" * 60)
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Model Loaded: {data.get('model_loaded', False)}")
            print(f"   Version: {data.get('version', 'unknown')}")
            print(f"   Predictions Served: {data.get('predictions_served', 0)}")
            if data.get('avg_inference_time'):
                print(f"   Avg Inference Time: {data.get('avg_inference_time'):.4f}s")
            print()
    except:
        pass

    # Next steps
    if passed == total:
        print("🚀 Next Steps:")
        print("-" * 60)
        print("   1. Test prediction with: python examples/api_usage_examples.py")
        print("   2. Run unit tests with: pytest tests/test_api.py -v")
        print("   3. Proceed to Docker containerization")
        print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        sys.exit(1)