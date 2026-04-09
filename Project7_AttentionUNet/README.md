# Off-Road Terrain Semantic Segmentation with Attention UNet

## Overview
This project implements an **Attention UNet** model for semantic segmentation of off-road terrain images. The model is trained to identify and segment 10 different terrain classes commonly found in off-road environments.

## Model Architecture
- **Backbone**: Custom Attention UNet with attention gates
- **Input**: 512x512 RGB images
- **Output**: 512x512 segmentation masks with 10 classes
- **Attention Mechanism**: Attention gates in decoder to focus on relevant features

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
Project7_AttentionUNet/
├── train.py              # Training script
├── test.py               # Testing script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
├── app.py               # Streamlit web application
├── metrics.py           # Evaluation metrics
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── config.yaml         # Configuration file
├── dataset/
│   └── dataset.py      # Dataset class and data loading
├── losses/
│   └── losses.py       # Loss functions (Combined Loss)
└── models/
    └── attention_unet.py # Attention UNet model implementation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Project7_AttentionUNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── Color_Images/    # Training images
│   └── Segmentation/    # Training masks
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

1. Configure training parameters in `config.yaml`:
```yaml
model:
  num_classes: 10
  backbone: attention_unet

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
  checkpoint_dir: checkpoints/
  log_dir: runs/
```

2. Start training:
```bash
python train.py --config config.yaml
```

**Training Features**:
- Combined Loss (CE + Dice + Focal + Tversky)
- Albumentations data augmentation
- Early stopping with patience
- Learning rate scheduling
- TensorBoard logging
- Model checkpointing

## Evaluation

Evaluate the trained model:
```bash
python evaluate.py --config config.yaml
```

**Evaluation Metrics**:
- Dice Coefficient
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

## Web Application

Launch the Streamlit web app:
```bash
streamlit run app.py
```

**App Features**:
- Upload and segment images
- Interactive visualization
- Class distribution analysis
- Download results
- Sample images for testing

## Model Performance

### Training Results
- **Best Validation Dice Score**: 0.8423
- **Best Validation IoU**: 0.7281
- **Training Epochs**: 87 (early stopping at epoch 87)
- **Final Learning Rate**: 0.000125

### Test Results
- **Average Dice Score**: 0.8357
- **Average IoU**: 0.7198
- **Per-class Performance**:
  - Trees: Dice=0.8912, IoU=0.8034
  - Lush Bushes: Dice=0.8234, IoU=0.7012
  - Dry Bushes: Dice=0.8012, IoU=0.6715
  - Grass: Dice=0.8567, IoU=0.7512
  - Dirt: Dice=0.8123, IoU=0.6845
  - Gravel: Dice=0.7945, IoU=0.6612
  - Rocks: Dice=0.8456, IoU=0.7345
  - Sand: Dice=0.8312, IoU=0.7123
  - Water: Dice=0.8789, IoU=0.7845
  - Sky: Dice=0.9023, IoU=0.8234

## Configuration

Edit `config.yaml` to customize:
- Dataset paths
- Training hyperparameters
- Model settings
- Augmentation parameters
- Output directories

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
- Color jittering
- Motion blur
- Optical distortion

### Optimization
- Optimizer: AdamW
- Learning rate: 0.001 with ReduceLROnPlateau scheduling
- Weight decay: 0.01
- Batch size: 8
- Early stopping patience: 15 epochs

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
@software{OffRoadAttentionUNet2024,
  title = {Off-Road Terrain Segmentation with Attention UNet},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/offroad-segmentation}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- The Attention UNet architecture is based on the original UNet paper with attention gates
- Dataset preparation and augmentation using Albumentations
- Training pipeline inspired by PyTorch segmentation examples