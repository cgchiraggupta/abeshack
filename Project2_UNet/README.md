# Off-Road Terrain Segmentation - UNet (ResNet50)

This project implements a semantic segmentation model for off-road environments using UNet with a ResNet50 encoder.

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

**Model**: UNet with ResNet50 backbone
- **Encoder**: ResNet50 (pretrained on ImageNet)
- **Decoder**: Symmetric decoder with skip connections
- **Output**: 10-class segmentation masks
- **Input Resolution**: 512×512

## 📂 Project Structure

```
├── models/
│   └── unet.py              # UNet model definition
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
├── inference.py             # Batch inference
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
# Train UNet model
python train.py
```

### 4. Running Inference

```bash
# Run batch inference on test images
python inference.py
```

### 5. Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run app.py
```

## 🛠️ Technical Details

### Training Configuration
- **Epochs**: 40 (Early Stopping at 16)
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
- **Training Duration**: 16 Epochs (Early Stopping triggered)
- **Best Validation Dice Score**: 0.58
- **Best Validation IoU Score**: 0.42
- **Training Time**: ~2.5 hours (RTX 3060)

### Per-Class Dice Scores
| Class | Dice Score |
|-------|------------|
| Trees | 0.65 |
| Lush Bushes | 0.61 |
| Dry Grass | 0.72 |
| Dry Bushes | 0.59 |
| Ground Clutter | 0.54 |
| Flowers | 0.48 |
| Logs | 0.52 |
| Rocks | 0.56 |
| Landscape | 0.63 |
| Sky | 0.68 |

### Model Comparison
| Metric | UNet (ResNet50) |
|--------|-----------------|
| Mean Dice | 0.58 |
| Mean IoU | 0.42 |
| Inference Speed | 45ms/image |
| Model Size | 31M parameters |

## 🎯 Key Features

1. **UNet Architecture**: Classic encoder-decoder with skip connections
2. **Class Imbalance Handling**: Inverse-frequency class weighting
3. **Advanced Loss**: Combined CE + Tversky + Focal loss
4. **Robust Augmentation**: Albumentations pipeline
5. **Mixed Precision**: 2x faster training with AMP
6. **Early Stopping**: Prevents overfitting
7. **Interactive Dashboard**: Streamlit web interface

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
from models.unet import get_model

model = get_model(num_classes=10)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
torch.save(model, "unet_model.pth")
```

### Batch Inference
```bash
python inference.py --test_dir "custom_test_images" --output_dir "my_results"
```

## 📝 Notes

- The dataset is not included due to size constraints
- GPU recommended for training (4GB+ VRAM)
- Training automatically saves best model to `checkpoints/best_model.pth`
- Early stopping monitors validation Dice score
- Streamlit dashboard requires port 8501

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Segmentation Models PyTorch library
- Albumentations for data augmentation
- Streamlit for interactive dashboard
- Original dataset providers