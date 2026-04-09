# Off-Road Terrain Segmentation - FCN ResNet50

This project implements a semantic segmentation model for off-road environments using Fully Convolutional Network (FCN) with ResNet50 backbone from torchvision.

## 📋 Project Overview

**Objective**: Semantic segmentation of off-road terrain images into 10 classes for autonomous vehicle navigation.

**Classes**:
- 0: Trees (100)
- 1: Lush Bushes (200)
- 2: Dry Grass (300)
- 3: Dry Bushes (500)
- 4: Ground Clutter (550)
- 5: Flowers (600)
- 6: Logs (700)
- 7: Rocks (800)
- 8: Landscape (7100)
- 9: Sky (10000)

## 🏗️ Model Architecture

**Model**: FCN with ResNet50 backbone
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Architecture**: Fully convolutional with atrous convolution
- **Parameters**: ~25 million
- **Input Resolution**: 512×512
- **Framework**: PyTorch torchvision

## 📂 Project Structure

```
├── models/
│   └── fcn.py               # FCN model definition
├── losses/
│   └── losses.py            # Combined loss functions
├── dataset/
│   └── dataset.py           # Dataset class with augmentations
├── checkpoints/             # Saved model weights
├── data/                    # Dataset (not included)
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   ├── val/
│   └── testImages/
├── train.py                 # Main training script
├── test.py                  # Model testing
├── evaluate.py              # Evaluation metrics
├── inference.py             # Batch inference (with timing)
├── app.py                   # Streamlit dashboard
├── metrics.py               # Dice & IoU metrics
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your dataset in the following structure:
```
data/
├── train/
│   ├── Color_Images/     # Training RGB images (*.png)
│   └── Segmentation/     # Training masks (*.png)
├── val/
│   ├── Color_Images/     # Validation RGB images
│   └── Segmentation/     # Validation masks
└── testImages/           # Test images for inference
```

### 3. Training the Model

```bash
# Train FCN model
python train.py
```

### 4. Running Inference

```bash
# Run batch inference on test images (with timing)
python inference.py
```

### 5. Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run app.py
```

## 🛠️ Technical Details

### Training Configuration
- **Epochs**: 40 (Early Stopping at 18)
- **Batch Size**: 4
- **Learning Rate**: 1e-4 (AdamW)
- **Weight Decay**: 1e-4
- **Mixed Precision**: Enabled (AMP)
- **Early Stopping Patience**: 8 epochs

### Loss Function
Combined Loss = CrossEntropy + Tversky + Focal
- **CrossEntropy**: Standard classification loss with class weights
- **Tversky**: α=0.7, β=0.3 (focus on false negatives)
- **Focal**: γ=2.0 (focus on hard examples)

### Class Weights
```
[50.4137, 50.4162, 2.9984, 50.4182, 50.418, 
 50.4183, 50.4183, 50.418, 16.598, 3.2536]
```

### Data Augmentation
- Horizontal/Vertical Flip
- Random Rotation (90°)
- Shift-Scale-Rotate
- Color Jitter
- Gaussian Blur
- Coarse Dropout
- Resize to 512×512

## 📊 Results

### Training Performance
- **Training Duration**: 18 Epochs (Early Stopping triggered)
- **Best Validation Dice Score**: 0.52
- **Best Validation IoU Score**: 0.36
- **Training Time**: ~2 hours (RTX 3060)

### Inference Performance
- **Average Inference Time**: 38ms per image (RTX 3060)
- **Fastest Inference**: 22ms
- **Slowest Inference**: 55ms
- **Memory Usage**: ~1.2GB VRAM

### Per-Class Dice Scores
| Class | Dice Score |
|-------|------------|
| Trees | 0.58 |
| Lush Bushes | 0.54 |
| Dry Grass | 0.65 |
| Dry Bushes | 0.52 |
| Ground Clutter | 0.48 |
| Flowers | 0.42 |
| Logs | 0.46 |
| Rocks | 0.50 |
| Landscape | 0.56 |
| Sky | 0.61 |

### Model Comparison
| Metric | FCN ResNet50 |
|--------|--------------|
| Mean Dice | 0.52 |
| Mean IoU | 0.36 |
| Inference Speed | 38ms/image |
| Model Size | 25M parameters |

## 🎯 Key Features

1. **FCN Architecture**: Classic fully convolutional network for segmentation
2. **Fast Inference**: Optimized for real-time applications
3. **Efficient Memory**: Low VRAM usage during inference
4. **Class Imbalance Handling**: Inverse-frequency class weighting
5. **Advanced Loss**: Combined CE + Tversky + Focal loss
6. **Robust Augmentation**: Albumentations pipeline
7. **Mixed Precision**: 2x faster training with AMP
8. **Early Stopping**: Prevents overfitting
9. **Interactive Dashboard**: Streamlit web interface
10. **Performance Monitoring**: Inference timing included

## 🔧 Advanced Usage

### Custom Training
Edit `train.py` to modify:
- Batch size
- Learning rate
- Number of epochs
- Class weights
- Augmentation strength

### Model Export
```python
import torch
from models.fcn import get_model

model = get_model(num_classes=10)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
torch.save(model, "fcn_model.pth")
```

### Batch Inference with Timing
```bash
python inference.py --test_dir "custom_test_images" --output_dir "my_results"
```

### Performance Benchmarking
The inference script automatically measures and reports:
- Average inference time
- Fastest inference time
- Slowest inference time

## 📝 Notes

- The dataset is not included due to size constraints
- GPU recommended for training (4GB+ VRAM)
- Training automatically saves best model to `checkpoints/best_model.pth`
- Early stopping monitors validation Dice score
- Streamlit dashboard requires port 8501
- FCN model is from torchvision (no external dependencies)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PyTorch torchvision library
- Albumentations for data augmentation
- Streamlit for interactive dashboard
- Original dataset providers