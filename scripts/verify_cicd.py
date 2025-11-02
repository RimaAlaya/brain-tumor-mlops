#!/usr/bin/env python3
"""
CI/CD Setup Verification Script
Usage: python scripts/verify_cicd.py
"""

import subprocess
import sys
from pathlib import Path


def check_passed(message):
    print(f"✅ {message}")


def check_failed(message):
    print(f"❌ {message}")
    return False


def check_warning(message):
    print(f"⚠️  {message}")


def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def main():
    print("=" * 60)
    print("🔄 CI/CD Setup Verification")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Check workflow files
    print("📁 Checking workflow files...")
    print("-" * 60)

    workflow_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/docker.yml",
        ".github/workflows/lint.yml"
    ]

    for file in workflow_files:
        if Path(file).exists():
            check_passed(f"{file} exists")
        else:
            check_failed(f"{file} is missing")
            all_passed = False

    print()

    # 2. Check config files
    print("⚙️  Checking configuration files...")
    print("-" * 60)

    config_files = [
        "setup.cfg",
        "pyproject.toml"
    ]

    for file in config_files:
        if Path(file).exists():
            check_passed(f"{file} exists")
        else:
            check_failed(f"{file} is missing")
            all_passed = False

    print()

    # 3. Check if tools are installed
    print("🔧 Checking development tools...")
    print("-" * 60)

    tools = {
        "pytest": "pytest --version",
        "black": "black --version",
        "isort": "isort --version",
        "flake8": "flake8 --version"
    }

    for tool_name, command in tools.items():
        success, stdout, stderr = run_command(command)
        if success:
            version = stdout.strip().split('\n')[0]
            check_passed(f"{tool_name} is installed ({version})")
        else:
            check_warning(f"{tool_name} not installed (optional for CI)")

    print()

    # 4. Test linting locally
    print("🧹 Testing local linting...")
    print("-" * 60)

    # Test black (dry run)
    success, _, _ = run_command("black --check src/ tests/ 2>/dev/null")
    if success:
        check_passed("Code formatting is correct (black)")
    else:
        check_warning("Code needs formatting (run: black src/ tests/)")

    # Test isort (dry run)
    success, _, _ = run_command("isort --check-only src/ tests/ 2>/dev/null")
    if success:
        check_passed("Imports are sorted correctly (isort)")
    else:
        check_warning("Imports need sorting (run: isort src/ tests/)")

    # Test flake8
    success, _, _ = run_command("flake8 src/ --count --select=E9,F63,F7,F82 2>/dev/null")
    if success:
        check_passed("No critical linting errors (flake8)")
    else:
        check_warning("Some linting errors found (check with: flake8 src/)")

    print()

    # 5. Check Git repository
    print("📦 Checking Git repository...")
    print("-" * 60)

    if Path(".git").exists():
        check_passed("Git repository initialized")

        # Check if on GitHub
        success, stdout, _ = run_command("git remote get-url origin 2>/dev/null")
        if success and "github.com" in stdout:
            check_passed("GitHub remote configured")
        else:
            check_warning("GitHub remote not configured")
    else:
        check_failed("Not a Git repository")
        all_passed = False

    print()

    # Summary
    print("=" * 60)
    print("📊 Verification Summary")
    print("=" * 60)
    print()

    if all_passed:
        check_passed("CI/CD setup is complete!")
        print()
        print("🚀 Next steps:")
        print("   1. Format code: black src/ tests/")
        print("   2. Sort imports: isort src/ tests/")
        print("   3. Run tests: pytest tests/ -v")
        print("   4. Commit: git add .github/ setup.cfg pyproject.toml")
        print("   5. Push: git push origin main")
        print("   6. Watch GitHub Actions: Check the 'Actions' tab")
        print()
        print("📖 For more info, see: CICD_SETUP_GUIDE.md")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        sys.exit(1)