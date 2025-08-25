## MemRoadNet
Code for our paper "MemRoadNet: Human-Like Memory Integration for Free Road Space Detection"

## 🚀 Overview

This repository contains the implementation of **MemRoadNet**, a novel approach for road segmentation that combines InternImage-XL backbone with UperNet decoder and an innovative human-like memory system. Our method achieves compettive performance on road segmentation tasks by leveraging episodic memory to improve segmentation consistency and accuracy.


## 📂 Repository Structure

```
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── internimage.py          # InternImage backbone implementation
│   │   ├── upernet.py             # UperNet decoder implementation
│   │   ├── memory_bank.py         # Human-like memory system
│   │   └── memory_augmented_model.py  # Complete model architecture
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration settings
│   │   ├── dataset.py             # Dataset loading and preprocessing
│   │   ├── losses.py              # Loss functions
│   │   └── metrics.py             # Evaluation metrics
│   └── data/
│       └── transforms.py          # Data augmentation and transforms
├── scripts/
│   ├── train.py                   # Training script
│   ├── inference.py               # Inference script
│   ├── evaluate.py                # Evaluation script
│   └── analyze_model.py           # Model analysis (FLOPs, params)
├── docs/
│   ├── model_analysis_report.md   # Detailed model analysis
│   └── training_guide.md          # Training instructions
├── requirements.txt               # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/memory-enhanced-internimage.git
cd memory-enhanced-internimage

# Create virtual environment
conda create -n memory-internimage python=3.8
conda activate memory-internimage

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Required Packages
```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy pandas
pip install tqdm scikit-learn
pip install albumentations
pip install timm
```

## 📊 Dataset

Our model is trained and evaluated on road segmentation datasets. The training data should be organized as follows:

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

### Supported Datasets:
- KITTI Road Dataset
- Virtual KITTI 2 Dataset
- R2D Dataset
- Cityscapes (road class)
- Custom road segmentation datasets

## 🚀 Quick Start

### Training

```bash
# Basic training
python scripts/train.py --config configs/default.yaml

# Training with custom settings
python scripts/train.py \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --memory_size 200 \
    --memory_weight 0.2
```

### Inference

```bash
# Single image inference
python scripts/inference.py \
    --model_path weights/best_model.pth \
    --input_path test_images/image.jpg \
    --output_path results/

# Batch inference
python scripts/inference.py \
    --model_path weights/best_model.pth \
    --input_dir test_images/ \
    --output_dir results/
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model_path weights/best_model.pth \
    --data_dir data/test/ \
    --output_file results.json
```

## 📁 Pretrained Models & Data

Download pretrained models, weights, and BEV visualization results from:

**🔗 [SharePoint Download Link](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/Eu2DbQrJ3RxBjzW7ov-ksqMBpP3E8CfLVMkH48YeT-254w?e=HjfzWk)**

Available downloads:
- Pretrained InternImage-XL weights
- Trained model checkpoints
- BEV (Bird's Eye View) visualization images
- Sample datasets and results
- Memory bank states

## 🔧 Configuration

Key configuration options in `src/utils/config.py`:

```python
# Model Configuration
IMAGE_SIZE = (640, 640)
BATCH_SIZE = 2
LEARNING_RATE = 0.0001

# Memory Configuration  
MEMORY_SIZE = 200
TOP_K_MEMORIES = 9
MEMORY_WEIGHT = 0.2
USE_MEMORY_IN_TRAINING = True

# Training Configuration
EPOCHS = 100
WEIGHTS_PATH = "./weights"
PRETRAINED_PATH = "pretrained/upernet_internimage_xl_640_160k_ade20k.pth"
```

## 📊 Model Analysis

Run model analysis to get detailed statistics:

```bash
python scripts/analyze_model.py
```

This will output:
- Parameter count and distribution
- FLOPs calculation
- Memory usage analysis
- Inference speed benchmarks

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex

```

## 🙏 Acknowledgments

- InternImage authors for the excellent backbone architecture
- UperNet authors for the segmentation head design
- The computer vision community for inspiration and feedback


## 🔗 Related Work

- [InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions](https://arxiv.org/abs/2211.05778)
- [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221)
- [Memory Networks for Visual Recognition](https://arxiv.org/abs/1511.06392)

---

⭐ **Star this repository if you find it useful!** ⭐
