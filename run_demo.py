#!/usr/bin/env python3
"""
Quick launcher for Gradio demo

Usage:
    python run_demo.py
    python run_demo.py --share  # Create public link
    python run_demo.py --port 8080  # Custom port
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.demo.gradio_app import create_demo
except ImportError as e:
    print(f"❌ Error importing demo: {e}")
    print("\n💡 Make sure you've installed gradio:")
    print("   pip install gradio")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio Demo")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed error messages"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🧠 Brain Tumor Classification - Gradio Demo")
    print("=" * 60)
    print()

    # Check if model exists
    models_dir = Path("models")
    model_keras = models_dir / "brain_tumor_model.keras"
    model_h5 = models_dir / "brain_tumor_model.h5"

    if not model_keras.exists() and not model_h5.exists():
        print("⚠️  WARNING: No model found in models/")
        print("   Please train the model first:")
        print("   python src/training/train.py")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    else:
        print("✅ Model found")

    print()
    print("Starting Gradio demo...")
    print(f"Port: {args.port}")
    print(f"Share: {'Yes' if args.share else 'No'}")
    print()

    # Create and launch demo
    demo = create_demo()

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            show_error=args.debug,
            show_api=False
        )
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()