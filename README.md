# Off-Road Terrain Semantic Segmentation - Multi-Model Implementation

This project implements multiple semantic segmentation models for off-road terrain analysis with 10 terrain classes. The project includes DeepLabV3+, UNet, PSPNet, and FPN architectures for comparative analysis.

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

## 🏗️ Model Architectures

1. **DeepLabV3+**: Baseline model with ResNet101 encoder
2. **UNet**: Classic encoder-decoder architecture with ResNet34 encoder
3. **PSPNet**: Pyramid Scene Parsing Network with ResNet50 encoder
4. **FPN**: Feature Pyramid Network with ResNet50 encoder

## 📂 Project Structure

```
├── models/                    # Model definitions
│   ├── deeplabv3plus.py      # DeepLabV3+ implementation
│   ├── unet.py               # UNet implementation
│   ├── pspnet.py             # PSPNet implementation
│   └── fpn.py                # FPN implementation
├── losses/                   # Loss functions
│   └── losses.py             # Combined loss (CE + Tversky + Focal)
├── dataset/                  # Data handling
│   └── dataset.py            # Dataset class with augmentations
├── checkpoints/              # Saved model weights
├── data/                     # Dataset (not included in repo)
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   ├── val/
│   └── testImages/
├── inference_results/        # Inference outputs
├── train.py                  # Main training script
├── train_unet.py             # UNet training script
├── train_pspnet.py           # PSPNet training script
├── train_fpn.py              # FPN training script
├── evaluate.py               # Model evaluation
├── inference.py              # Batch inference
├── app.py                    # Streamlit dashboard
├── metrics.py                # Evaluation metrics
├── early_stopping.py         # Early stopping implementation
├── compute_weights.py        # Class weight computation
├── check_leakage.py          # Dataset integrity check
├── requirements.txt          # Dependencies
└── README.md                 # This file
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

### 3. Training Models

```bash
# Train DeepLabV3+ (baseline)
python train.py

# Train UNet
python train_unet.py

# Train PSPNet
python train_pspnet.py

# Train FPN
python train_fpn.py
```

### 4. Running Inference

```bash
# Run batch inference on test images
python inference.py --model deeplabv3plus  # or unet, pspnet, fpn
```

### 5. Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run app.py
```

## 🛠️ Technical Details

### Model Configurations

- **Input Size**: 512×512 RGB images
- **Encoder**: Pretrained on ImageNet
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 1e-4 with ReduceLROnPlateau scheduling
- **Batch Size**: 4 (adjust based on GPU memory)
- **Loss Function**: Combined Loss (CrossEntropy + Tversky + Focal)
- **Class Weights**: Inverse frequency weighting to handle imbalance

### Training Features

- **Mixed Precision Training (AMP)**: 2x faster training with reduced memory
- **Early Stopping**: Prevents overfitting based on validation Dice score
- **Data Augmentation**: Robust pipeline using Albumentations
- **Encoder Freezing**: Stabilizes training by freezing encoder for first 5 epochs
- **Class Weighting**: Addresses extreme class imbalance

### Evaluation Metrics

- **Dice Score (F1-Score)**: Primary metric for segmentation quality
- **IoU (Jaccard Index)**: Intersection over Union
- **Per-Class Metrics**: Individual class performance analysis
- **Visualization**: Side-by-side comparison of predictions

## 📊 Performance Comparison

| Model | Encoder | Parameters | Inference Speed | Best Dice |
|-------|---------|------------|-----------------|-----------|
| DeepLabV3+ | ResNet101 | ~60M | Medium | 0.60 |
| UNet | ResNet34 | ~25M | Fast | TBD |
| PSPNet | ResNet50 | ~50M | Medium | TBD |
| FPN | ResNet50 | ~35M | Fast | TBD |

## 🎯 Key Features

1. **Multi-Model Support**: Compare different segmentation architectures
2. **Class Imbalance Handling**: Advanced weighting and loss functions
3. **Interactive Dashboard**: Real-time inference with Streamlit
4. **Production Ready**: Includes training, evaluation, and inference pipelines
5. **Extensible Design**: Easy to add new models or datasets

## 🔧 Advanced Usage

### Custom Training Configuration

Edit training parameters in individual training scripts:
- Batch size
- Learning rate
- Number of epochs
- Class weights
- Augmentation strength

### Adding New Models

1. Create model definition in `models/` directory
2. Add training script following existing patterns
3. Update `app.py` to support new model

### Exporting Models

```python
import torch
from models.deeplabv3plus import get_model

model = get_model(num_classes=10)
model.load_state_dict(torch.load("checkpoints/best_model_deeplabv3plus.pth"))
torch.save(model, "exported_model.pth")
```

## 📝 Notes

- The dataset is not included due to size constraints
- Pretrained weights are available in `checkpoints/` after training
- GPU recommended for training (4GB+ VRAM)
- Training time: ~2-4 hours per model on RTX 3060

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Segmentation Models PyTorch library
- Albumentations for data augmentation
- Streamlit for interactive dashboard
- Original dataset providers