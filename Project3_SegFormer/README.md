# Off-Road Terrain Segmentation - SegFormer-B2

This project implements a semantic segmentation model for off-road environments using SegFormer-B2 from HuggingFace Transformers.

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

**Model**: SegFormer-B2 (Transformer-based)
- **Backbone**: Mix Transformer (MiT-B2)
- **Architecture**: Hierarchical encoder with lightweight MLP decoder
- **Parameters**: ~27 million
- **Input Resolution**: 512×512
- **Framework**: HuggingFace Transformers

## 📂 Project Structure

```
├── models/
│   └── segformer.py          # SegFormer model definition
├── losses/
│   └── losses.py             # Combined loss functions
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
# Train SegFormer-B2 model
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
- **Epochs**: 40 (Early Stopping at 12)
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
- **Training Duration**: 12 Epochs (Early Stopping triggered)
- **Best Validation Dice Score**: 0.63
- **Best Validation IoU Score**: 0.47
- **Training Time**: ~3 hours (RTX 3060)

### Per-Class Dice Scores
| Class | Dice Score |
|-------|------------|
| Trees | 0.68 |
| Lush Bushes | 0.64 |
| Dry Grass | 0.75 |
| Dry Bushes | 0.62 |
| Ground Clutter | 0.57 |
| Flowers | 0.51 |
| Logs | 0.55 |
| Rocks | 0.59 |
| Landscape | 0.66 |
| Sky | 0.71 |

### Model Comparison
| Metric | SegFormer-B2 |
|--------|--------------|
| Mean Dice | 0.63 |
| Mean IoU | 0.47 |
| Inference Speed | 52ms/image |
| Model Size | 27M parameters |

## 🎯 Key Features

1. **Transformer Architecture**: State-of-the-art vision transformer for segmentation
2. **Hierarchical Encoder**: Multi-scale feature extraction
3. **Lightweight MLP Decoder**: Efficient decoding without complex operations
4. **Class Imbalance Handling**: Inverse-frequency class weighting
5. **Advanced Loss**: Combined CE + Tversky + Focal loss
6. **Robust Augmentation**: Albumentations pipeline
7. **Mixed Precision**: 2x faster training with AMP
8. **Early Stopping**: Prevents overfitting
9. **Interactive Dashboard**: Streamlit web interface

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
from models.segformer import get_model

model = get_model(num_classes=10)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
torch.save(model, "segformer_model.pth")
```

### Batch Inference
```bash
python inference.py --test_dir "custom_test_images" --output_dir "my_results"
```

## 📝 Notes

- The dataset is not included due to size constraints
- GPU recommended for training (6GB+ VRAM for transformers)
- Training automatically saves best model to `checkpoints/best_model.pth`
- Early stopping monitors validation Dice score
- Streamlit dashboard requires port 8501
- SegFormer requires HuggingFace transformers library

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- HuggingFace Transformers library
- NVIDIA for SegFormer implementation
- Albumentations for data augmentation
- Streamlit for interactive dashboard
- Original dataset providers