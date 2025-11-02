#!/usr/bin/env python3
"""
Docker Setup Verification Script (Cross-platform)
Usage: python scripts/verify_docker.py
"""

import subprocess
import sys
from pathlib import Path
import os


class Colors:
    """ANSI color codes"""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def is_windows():
        return os.name == 'nt'

    @classmethod
    def disable_on_windows(cls):
        """Disable colors on Windows if not supported"""
        if cls.is_windows():
            cls.GREEN = cls.RED = cls.YELLOW = cls.CYAN = cls.NC = ''


# Disable colors on Windows for better compatibility
Colors.disable_on_windows()


def check_passed(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")


def check_failed(message):
    """Print failure message"""
    print(f"{Colors.RED}❌ {message}{Colors.NC}")


def check_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")


def run_command(command, capture=True):
    """Run a shell command and return success status"""
    try:
        if capture:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        else:
            result = subprocess.run(command, shell=True, timeout=30)
            return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)


def main():
    """Main verification function"""
    print("=" * 60)
    print(f"{Colors.CYAN}🐳 Docker Setup Verification{Colors.NC}")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Check Docker is installed
    print(f"{Colors.CYAN}📋 Checking prerequisites...{Colors.NC}")
    print("-" * 60)

    success, output = run_command("docker --version")
    if success:
        check_passed(f"Docker is installed ({output})")
    else:
        check_failed("Docker is not installed")
        print("Install from: https://docs.docker.com/get-docker/")
        all_passed = False

    success, output = run_command("docker-compose --version")
    if success:
        check_passed(f"Docker Compose is installed ({output})")
    else:
        check_failed("Docker Compose is not installed")
        all_passed = False

    # 2. Check Docker is running
    success, _ = run_command("docker info")
    if success:
        check_passed("Docker daemon is running")
    else:
        check_failed("Docker daemon is not running")
        print("Start Docker Desktop or run: sudo systemctl start docker")
        all_passed = False

    print()

    # 3. Check required files
    print(f"{Colors.CYAN}📁 Checking Docker files...{Colors.NC}")
    print("-" * 60)

    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "docker/Dockerfile.train",
        "docker/Dockerfile.serve"
    ]

    for file in required_files:
        if Path(file).exists():
            check_passed(f"{file} exists")
        else:
            check_failed(f"{file} is missing")
            all_passed = False

    print()

    # 4. Check model exists
    print(f"{Colors.CYAN}🤖 Checking model files...{Colors.NC}")
    print("-" * 60)

    model_exists = (
            Path("models/brain_tumor_model.keras").exists() or
            Path("models/brain_tumor_model.h5").exists()
    )

    if model_exists:
        check_passed("Model file found")
    else:
        check_warning("Model file not found in models/")
        print("   Train model first: docker-compose --profile training run --rm train")

    if Path("models/class_names.json").exists():
        check_passed("Class names file found")
    else:
        check_warning("class_names.json not found")

    print()

    # 5. Try to build images (optional - skipped for speed)
    print(f"{Colors.CYAN}🔨 Docker build capability...{Colors.NC}")
    print("-" * 60)

    # Just verify Dockerfiles are valid, don't actually build
    if Path("Dockerfile").exists():
        check_passed("Dockerfile is present (build skipped for speed)")
        print("   To build: docker build -t brain-tumor-api .")

    print()

    # 6. Summary
    print("=" * 60)
    print(f"{Colors.CYAN}📊 Verification Summary{Colors.NC}")
    print("=" * 60)
    print()

    if all_passed:
        check_passed("Docker setup is complete!")
        print()
        print(f"{Colors.CYAN}🚀 Next steps:{Colors.NC}")
        print("   1. Start API: docker-compose up -d api")
        print("   2. Check health: curl http://localhost:8000/health")
        print("   3. View logs: docker-compose logs -f api")
        print("   4. Run training: docker-compose --profile training run --rm train")
        print("   5. View docs: http://localhost:8000/docs")
        print()
        print(f"📖 For more info, see: DOCKER_GUIDE.md")
        return 0
    else:
        print(f"{Colors.YELLOW}⚠️  Some checks failed. Please fix the issues above.{Colors.NC}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Verification interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}❌ Unexpected error: {str(e)}{Colors.NC}")
        sys.exit(1)