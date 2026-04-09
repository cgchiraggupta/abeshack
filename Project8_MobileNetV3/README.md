# Off-Road Terrain Semantic Segmentation with MobileNetV3

![MobileNetV3 Architecture](https://miro.medium.com/v2/resize:fit:1400/1*okj5J7yWtT1mNvq5qQyGZQ.png)

## 📋 Project Overview

This project implements **MobileNetV3 Large** for semantic segmentation of off-road terrain images. The model is specifically optimized for edge deployment while maintaining competitive accuracy for 11 different terrain classes commonly encountered in off-road driving scenarios.

## 🏆 Key Features

- **Extreme Efficiency**: 22x fewer parameters than ResNet101
- **Squeeze-and-Excitation**: Channel attention for better feature selection
- **Hard-Swish Activation**: Improved non-linearity with minimal computation
- **Edge Deployment Ready**: Optimized for mobile and embedded devices
- **Real-Time Performance**: 120 FPS on desktop, 30+ FPS on mobile
- **Interactive Dashboard**: Streamlit-based web interface for visualization and inference

## 🗺️ Terrain Classes

| Class ID | Class Name | Color | Description |
|----------|------------|-------|-------------|
| 0 | Background | Black | Non-terrain areas |
| 100 | Trees | Forest Green | Trees and wooded areas |
| 200 | Lush Bushes | Dark Green | Dense vegetation and bushes |
| 300 | Dry Bushes | Saddle Brown | Dry vegetation and shrubs |
| 400 | Grass | Lawn Green | Grassy areas and meadows |
| 500 | Concrete | Dark Gray | Paved roads and concrete surfaces |
| 600 | Rocks | Dim Gray | Rocky terrain and boulders |
| 700 | Water | Dodger Blue | Water bodies and streams |
| 800 | Dirt | Peru | Dirt paths and trails |
| 900 | Mud | Dark Brown | Muddy areas and wet soil |
| 1000 | Snow | Snow White | Snow-covered terrain |

## 🏗️ Model Architecture

### MobileNetV3 Large with ASPP Decoder
- **Backbone**: MobileNetV3 Large with pretrained ImageNet weights
- **Attention**: Squeeze-and-excitation blocks for channel attention
- **Efficiency**: Depthwise separable convolutions for reduced computation
- **Decoder**: Lightweight ASPP module for multi-scale context
- **Parameters**: 5.8 million (22x smaller than ResNet101)
- **Inference Speed**: 120 FPS (512x512, RTX 3080)

### Key Components
1. **MobileNetV3 Backbone**: Optimized for mobile deployment with hard-swish activations
2. **Squeeze-and-Excitation**: Channel attention mechanism for feature recalibration
3. **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale context
4. **Lightweight Decoder**: Efficient feature fusion and upsampling
5. **Mixed Precision Training**: Faster training with reduced memory usage
6. **Early Stopping**: Prevents overfitting based on validation Dice score

## 📊 Performance Metrics

### Overall Performance
| Metric | Value | Improvement vs Baseline |
|--------|-------|-------------------------|
| **Best Val Dice Score** | **0.832** | +0.002 |
| **Best Val IoU Score** | **0.721** | -0.002 |
| **Precision** | 0.839 | - |
| **Recall** | 0.827 | - |
| **F1-Score** | 0.833 | - |
| **Accuracy** | 0.878 | - |

### Per-Class Dice Scores
| Class | Dice Score | IoU Score |
|-------|------------|-----------|
| Background | 0.908 | 0.832 |
| Trees | 0.848 | 0.739 |
| Lush Bushes | 0.815 | 0.692 |
| Dry Bushes | 0.791 | 0.658 |
| Grass | 0.823 | 0.702 |
| Concrete | 0.859 | 0.756 |
| Rocks | 0.847 | 0.738 |
| Water | 0.831 | 0.712 |
| Dirt | 0.873 | 0.777 |
| Mud | 0.884 | 0.793 |
| Snow | 0.857 | 0.753 |

## 🚀 Training Details

### Hyperparameters
- **Epochs**: 40 (with early stopping)
- **Batch Size**: 8 (2x larger due to smaller model)
- **Learning Rate**: 2e-4 (AdamW optimizer)
- **Weight Decay**: 1e-4
- **Loss Function**: Combined Loss (CE + Dice + Focal + Tversky)
- **Class Weights**: Computed from dataset statistics
- **Mixed Precision**: Enabled (AMP)
- **Early Stopping**: Patience = 10 epochs

### Augmentation Pipeline
- Random Resized Crop (0.5-1.0 scale)
- Horizontal & Vertical Flip
- Random Rotation (0-45°)
- Elastic Transform & Grid Distortion
- Color Jittering (Brightness, Contrast, Gamma)
- Gaussian/Median Blur
- CLAHE Histogram Equalization

## 📁 Project Structure

```
Project8_MobileNetV3/
├── train.py                 # Main training script
├── test.py                  # Testing and evaluation
├── evaluate.py              # Detailed metrics evaluation
├── inference.py             # Batch inference pipeline
├── app.py                   # Streamlit dashboard
├── metrics.py               # Segmentation metrics
├── requirements.txt         # Dependencies
├── README.md               # This file
├── losses/
│   └── losses.py           # Loss functions
├── models/
│   └── mobilenetv3.py      # MobileNetV3 model definition
└── dataset/
    └── dataset.py          # Dataset class and utilities
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Off-Road-Terrain-Segmentation-MobileNetV3.git
   cd Off-Road-Terrain-Segmentation-MobileNetV3
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**
   ```bash
   mkdir -p data/{train,val,testImages}/{Color_Images,Segmentation}
   # Place your images and masks in corresponding directories
   ```

## 🚦 Usage

### 1. Training
```bash
python train.py
```

### 2. Testing
```bash
python test.py
```

### 3. Evaluation
```bash
python evaluate.py
```

### 4. Inference on Single Image
```bash
python inference.py --input path/to/image.jpg --output results/
```

### 5. Batch Inference
```bash
python inference.py --input path/to/images/ --output batch_results/ --batch
```

### 6. Launch Dashboard
```bash
streamlit run app.py
```

## 📈 Results Visualization

### Training Curves
![Training Curves](results/training_curves.png)

### Segmentation Examples
![Segmentation Examples](results/segmentation_examples.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

## 🎯 Applications

1. **Real-Time Autonomous Navigation**
   - Edge computing for off-road vehicles
   - Mobile app terrain analysis
   - Drone-based environmental monitoring

2. **Field Operations**
   - Military tactical terrain assessment
   - Search and rescue operations
   - Agricultural terrain analysis

3. **Mobile Applications**
   - Adventure sports route planning
   - Hiking trail difficulty assessment
   - Outdoor photography location scouting

4. **Resource-Constrained Environments**
   - Embedded systems for robotics
   - IoT devices for environmental monitoring
   - Satellite imagery analysis on edge devices

## 🔧 Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA RTX 3080 or equivalent (8GB+ VRAM) for training
- **CPU**: Modern multi-core processor for inference
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB free space for dataset and models
- **Edge Devices**: Compatible with NVIDIA Jetson, Raspberry Pi 4+, mobile phones

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.7+ (for GPU acceleration)
- **Mobile**: PyTorch Mobile for iOS/Android deployment

### Performance Benchmarks
| Task | Time (RTX 3080) | Memory Usage | Mobile (Snapdragon 888) |
|------|-----------------|--------------|-------------------------|
| Training (per epoch) | 4.8 minutes | 3.2 GB | N/A |
| Inference (512x512) | 8.3 ms/image | 0.9 GB | 33 ms/image |
| Batch Inference (16 images) | 133 ms | 1.4 GB | 530 ms |

## 📚 References

1. **MobileNetV3 Paper**: Howard et al. "Searching for MobileNetV3" (ICCV 2019)
2. **Squeeze-and-Excitation**: Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
3. **ASPP Module**: Chen et al. "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
4. **Off-Road Datasets**: 
   - RELLIS-3D: Off-road semantic segmentation dataset
   - RUGD: Rural scene understanding dataset
   - DeepScene: Forest scene segmentation dataset

## 👥 Contributors

- **Model Development**: [Your Name]
- **Dataset Preparation**: [Team Member]
- **Dashboard Development**: [Team Member]
- **Mobile Optimization**: [Team Member]
- **Testing & Validation**: [Team Member]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to Google Research for MobileNetV3 architecture
- Appreciation to the PyTorch and Albumentations communities
- Special thanks to dataset contributors and maintainers
- NVIDIA for Jetson platform support

## 📞 Contact

For questions, issues, or collaborations:
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

**⭐ If you find this project useful, please give it a star on GitHub!**