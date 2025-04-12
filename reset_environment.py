#!/usr/bin/env python
"""
Environment Reset Helper

This script helps fix common dependency conflicts by uninstalling problematic packages
and reinstalling them with the correct versions.
"""

import subprocess
import sys
import os
import platform

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def reset_environment():
    """Reset the Python environment to fix dependency conflicts"""
    print("\n=== Environment Reset Helper ===\n")
    
    # Check platform
    system = platform.system()
    print(f"Detected platform: {system}")
    
    # Get Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")
    
    # Fix Pillow version conflict
    print("\n1. Fixing Pillow version conflict...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "pillow"])
    run_command([sys.executable, "-m", "pip", "install", "pillow>=10.1.0"])
    
    # Fix transformers/sentence-transformers conflict
    print("\n2. Reinstalling transformers and sentence-transformers...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "transformers", "sentence-transformers"])
    run_command([sys.executable, "-m", "pip", "install", "transformers==4.38.2", "sentence-transformers==2.3.1"])
    
    # Check if FAISS is needed
    if system != "Darwin":  # Not Mac
        print("\n3. Attempting to install FAISS for your platform...")
        faiss_installed = run_command([sys.executable, "-m", "pip", "install", "faiss-cpu>=1.7.4"])
        if not faiss_installed:
            print("FAISS installation failed, but the application will still work with the numpy fallback.")
    else:
        print("\n3. FAISS installation on Mac requires special handling.")
        print("The application will use the numpy fallback automatically.")
        print("If you want to try installing FAISS on Mac, you can try:")
        print("   conda install -c conda-forge faiss-cpu")
    
    print("\n=== Environment Reset Complete ===")
    print("\nYou can now try running the application with:")
    print("   python run_app.py")

if __name__ == "__main__":
    reset_environment() 