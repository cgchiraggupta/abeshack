#!/usr/bin/env python3
"""
Setup script for Off-Road Terrain Segmentation project.
This script helps set up the project environment and directory structure.
"""

import os
import sys
import subprocess
import argparse

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print(f"❌ Python 3.10 or higher is required. Current version: {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    return True

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "checkpoints",
        "inference_results",
        "logs",
        "assets",
        "data/train/Color_Images",
        "data/train/Segmentation",
        "data/val/Color_Images",
        "data/val/Segmentation",
        "data/testImages",
        "models",
        "losses",
        "dataset"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directory structure created")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_sample_data():
    """Download sample data if needed"""
    print("\nSample Data Setup:")
    print("="*50)
    print("The dataset is not included in the repository due to size constraints.")
    print("\nPlease ensure your dataset follows this structure:")
    print("""
data/
├── train/
│   ├── Color_Images/     # Training RGB images (*.png)
│   └── Segmentation/     # Training masks (*.png)
├── val/
│   ├── Color_Images/     # Validation RGB images
│   └── Segmentation/     # Validation masks
└── testImages/           # Test images for inference
    """)
    
    print("\nYou can:")
    print("1. Place your own dataset in the 'data/' directory")
    print("2. Use the sample images in the 'assets/' directory for testing")
    print("3. Contact the project maintainer for dataset access")
    
    return True

def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  No GPU detected. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check skipped.")
        return False

def create_sample_config():
    """Create sample configuration files"""
    print("\nCreating sample configuration...")
    
    sample_images = [
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        "https://images.unsplash.com/photo-1519681393784-d120267933ba",
        "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07"
    ]
    
    print("Sample images references added to assets/README.md")
    
    with open("assets/README.md", "w") as f:
        f.write("# Sample Images\n\n")
        f.write("Place sample images here for testing the application.\n\n")
        f.write("Suggested sources:\n")
        for url in sample_images:
            f.write(f"- {url}\n")
    
    return True

def run_quick_test():
    """Run a quick test to verify setup"""
    print("\nRunning quick test...")
    
    try:
        import torch
        import numpy as np
        
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
        
        test_tensor = torch.randn(2, 3, 512, 512)
        print(f"✅ Tensor operations working: {test_tensor.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup Off-Road Terrain Segmentation project')
    parser.add_argument('--skip-install', action='store_true', help='Skip package installation')
    parser.add_argument('--skip-gpu-check', action='store_true', help='Skip GPU check')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test after setup')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Off-Road Terrain Segmentation - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            print("⚠️  Dependency installation failed. You may need to install manually.")
    
    # Check GPU
    if not args.skip_gpu_check:
        check_gpu_availability()
    
    # Download sample data info
    download_sample_data()
    
    # Create sample config
    create_sample_config()
    
    # Run quick test
    if args.quick_test:
        run_quick_test()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your dataset in the 'data/' directory")
    print("2. Train a model: python train.py")
    print("3. Run inference: python inference.py")
    print("4. Launch dashboard: streamlit run app.py")
    print("\nFor help: python train.py --help")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())