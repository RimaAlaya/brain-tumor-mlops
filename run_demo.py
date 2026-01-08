#!/usr/bin/env python3
"""
Gradio Demo for Brain Tumor Classification
Render.com deployment friendly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.demo.gradio_app import create_demo
except ImportError as e:
    print(f"❌ Error importing demo: {e}")
    sys.exit(1)


def main():
    print("=" * 60)
    print("🧠 Brain Tumor Classification - Gradio Demo")
    print("=" * 60)

    # Check if model exists
    models_dir = Path("models")
    model_keras = models_dir / "brain_tumor_model.keras"
    model_h5 = models_dir / "brain_tumor_model.h5"

    if not model_keras.exists() and not model_h5.exists():
        print("⚠️  WARNING: No model found!")
        print("   The demo will still work but predictions may fail.")
    else:
        print("✅ Model found")

    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 7860))

    print(f"\n🚀 Starting Gradio on port {port}...")

    # Create and launch demo
    demo = create_demo()

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,  # Don't need share on Render
            show_error=True,
            show_api=False
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()