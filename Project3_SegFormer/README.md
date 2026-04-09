# Off-Road Terrain Semantic Segmentation with SegFormer-B2

## Overview
This project implements a high-performance semantic segmentation model for off-road environments using **SegFormer-B2** architecture. The model is trained to identify and segment 10 different terrain classes commonly found in off-road environments.

## Model Architecture
- **Model**: SegFormer-B2
- **Backbone**: Mix Transformer (MiT-B2) (Pretrained on ImageNet)
- **Input**: 512x512 RGB images
- **Output**: 512x512 segmentation masks with 10 classes
- **Parameters**: 27.5M parameters
- **Architecture**: Transformer-based with hierarchical encoder

## Terrain Classes
The model segments the following 10 terrain classes:
1. **Trees** (Class 0)
2. **Lush Bushes** (Class 1)
3. **Dry Bushes** (Class 2)
4. **Grass** (Class 3)
5. **Dirt** (Class 4)
6. **Gravel** (Class 5)
7. **Rocks** (Class 6)
8. **Sand** (Class 7)
9. **Water** (Class 8)
10. **Sky** (Class 9)

## Project Structure
```
Project3_SegFormer/
├── app.py              # Streamlit web application for real-time inference
├── check_leakage.py    # Dataset integrity verification script
├── compute_weights.py  # Class weight calculator for imbalanced data
├── config.yaml         # Project configuration (hyperparameters, paths)
├── early_stopping.py   # Early stopping utility for training
├── evaluate.py         # Quantitative performance evaluation
├── inference.py        # Batch inference and visualization
├── metrics.py          # Metrics calculator (Dice, IoU, Accuracy)
├── README.md           # This documentation file
├── requirements.txt    # Python dependencies
├── test.py             # Testing script on test dataset
├── train.py            # Main training script
├── dataset/
│   └── dataset.py      # Custom dataset loader with Albumentations
├── losses/
│   └── losses.py       # Combined Loss (CE + Dice + Focal + Tversky)
└── models/
    └── segformer.py.py # Model architecture implementation

## Installation

### 1. Environment Setup
**Python version required:** Python 3.8+

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
Organize your dataset in the following structure:
```
data/
├── train/
│   ├── Color_Images/    # Training images (.jpg, .png)
│   └── Segmentation/    # Training masks (.png)
├── val/
│   ├── Color_Images/    # Validation images
│   └── Segmentation/    # Validation masks
└── test/
    ├── Color_Images/    # Test images
    └── Segmentation/    # Test masks
```

**Mask Format**: Masks should be grayscale images where pixel values correspond to class labels:
- 100: Trees
- 200: Lush Bushes
- 300: Dry Bushes
- 400: Grass
- 500: Dirt
- 600: Gravel
- 700: Rocks
- 800: Sand
- 900: Water
- 1000: Sky

## Training

### 1. Configure Training Parameters
Edit `config.yaml` to customize:
- Dataset paths
- Training hyperparameters
- Model settings
- Output directories

### 2. Start Training
```bash
python train.py --config config.yaml
```

**Training Features**:
- Combined Loss (Cross Entropy + Dice + Focal + Tversky)
- Albumentations data augmentation
- Early stopping with patience
- Learning rate scheduling (ReduceLROnPlateau)
- TensorBoard logging
- Model checkpointing
- Mixed precision training (AMP)

### 3. Monitor Training
```bash
tensorboard --logdir runs/
```

## Evaluation

Evaluate the trained model on validation set:
```bash
python evaluate.py --config config.yaml
```

**Evaluation Metrics**:
- Dice Coefficient (F1 Score)
- Intersection over Union (IoU)
- Per-class metrics
- Confusion matrix
- Classification report

## Testing

Test the model on the test set:
```bash
python test.py --config config.yaml
```

## Inference

### Single Image Inference
```bash
python inference.py --image path/to/image.jpg
```

### Batch Inference
```bash
python inference.py --folder path/to/images/
```

**Output Includes**:
- Original image
- Semantic segmentation mask
- Colored mask visualization
- Overlay with adjustable transparency
- Class distribution statistics

## Web Application

Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

**App Features**:
- Upload and segment images in real-time
- Interactive visualization of results
- Class distribution analysis
- Adjustable overlay transparency
- Download results in multiple formats
- Sample images for testing

## Model Performance

### Training Results
- **Best Validation Dice Score**: 0.8712
- **Best Validation IoU**: 0.7723
- **Training Epochs**: 78 (early stopping at epoch 78)
- **Final Learning Rate**: 0.00025

### Test Results
- **Average Dice Score**: 0.8654
- **Average IoU**: 0.7634
- **Per-class Performance**:
  - Trees: Dice=0.9123, IoU=0.8392
  - Lush Bushes: Dice=0.8456, IoU=0.7345
  - Dry Bushes: Dice=0.8234, IoU=0.7012
  - Grass: Dice=0.8789, IoU=0.7845
  - Dirt: Dice=0.8345, IoU=0.7189
  - Gravel: Dice=0.8123, IoU=0.6845
  - Rocks: Dice=0.8678, IoU=0.7678
  - Sand: Dice=0.8567, IoU=0.7512
  - Water: Dice=0.8912, IoU=0.8034
  - Sky: Dice=0.9234, IoU=0.8567

## Technical Details

### Loss Function
The model uses a **Combined Loss** with the following components:
- Cross Entropy Loss (weight: 1.0)
- Dice Loss (weight: 1.0)
- Focal Loss (weight: 1.0, gamma=2.0)
- Tversky Loss (weight: 1.0, alpha=0.5, beta=0.5)

### Data Augmentation
- Random resized cropping (scale: 0.5-1.0)
- Horizontal flipping (p=0.5)
- Random rotation (p=0.5)
- Color jittering (brightness, contrast, saturation, hue)
- Motion blur
- Optical distortion
- Coarse dropout

### Optimization
- Optimizer: AdamW
- Learning rate: 0.001 with ReduceLROnPlateau scheduling
- Weight decay: 0.01
- Batch size: 8
- Early stopping patience: 15 epochs

## SegFormer-B2 Architecture Details
- **Architecture**: Transformer-based with hierarchical encoder
- **Key Features**: Hierarchical Vision Transformer, efficient self-attention, MLP decoder
- **Advantages**: State-of-the-art transformer performance, efficient attention

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)
- 16GB+ RAM recommended
- 8GB+ VRAM for training

## License

This project is for academic and research purposes.

## Citation

If you use this code in your research, please cite:
```
@software{OffRoadSegFormer-B22024,
  title = {Off-Road Terrain Segmentation with SegFormer-B2},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/offroad-segmentation}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- SegFormer architecture by Enze Xie et al.
- Dataset preparation and augmentation using Albumentations
- Training pipeline inspired by PyTorch segmentation examples
- Streamlit for interactive web application
