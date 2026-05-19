#!/usr/bin/env python3
"""
Build and publish revive-my-lover to PyPI.

Usage:
    python build_and_publish.py          # Build only
    python build_and_publish.py --test   # Upload to TestPyPI
    python build_and_publish.py --prod   # Upload to PyPI
"""

import os
import sys
import subprocess
import shutil


def run(cmd, check=True):
    """Run a command."""
    print(f"📦 {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    return result


def clean():
    """Clean build artifacts."""
    dirs_to_clean = ["build", "dist", "src/*.egg-info"]
    for pattern in dirs_to_clean:
        for path in [pattern]:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"🧹 Cleaned {path}")


def build():
    """Build the package."""
    print("\n🔨 Building package...")
    run("python -m build")
    
    # Check what was built
    dist_files = os.listdir("dist")
    print(f"\n✅ Built files:")
    for f in dist_files:
        size = os.path.getsize(os.path.join("dist", f))
        print(f"   {f} ({size / 1024:.1f} KB)")


def check():
    """Check the package."""
    print("\n🔍 Checking package...")
    run("twine check dist/*")


def upload_test():
    """Upload to TestPyPI."""
    print("\n🚀 Uploading to TestPyPI...")
    run("twine upload --repository testpypi dist/*")


def upload_prod():
    """Upload to PyPI."""
    print("\n🚀 Uploading to PyPI...")
    run("twine upload dist/*")


def main():
    """Main entry point."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("💘 revive-my-lover PyPI Build Script")
    print("=" * 60)
    
    # Check if build tools are installed
    try:
        import build
        import twine
    except ImportError:
        print("📦 Installing build tools...")
        run("pip install build twine -q")
    
    # Clean
    clean()
    
    # Build
    build()
    
    # Check
    check()
    
    # Upload based on argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            upload_test()
            print("\n✅ Uploaded to TestPyPI!")
            print("   Test with: pip install --index-url https://test.pypi.org/simple/ revive-my-lover")
        elif sys.argv[1] == "--prod":
            confirm = input("\n⚠️  Upload to PRODUCTION PyPI? (yes/no): ")
            if confirm.lower() == "yes":
                upload_prod()
                print("\n✅ Uploaded to PyPI!")
                print("   Install with: pip install revive-my-lover")
            else:
                print("❌ Cancelled")
        else:
            print(f"\n❓ Unknown argument: {sys.argv[1]}")
            print("   Use: --test or --prod")
    else:
        print("\n✅ Build complete! Files in dist/")
        print("   Next steps:")
        print("   1. python build_and_publish.py --test  # Test on TestPyPI")
        print("   2. python build_and_publish.py --prod  # Publish to PyPI")


if __name__ == "__main__":
    main()
