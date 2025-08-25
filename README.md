## MemRoadNet
Code for our paper "MemRoadNet: Human-Like Memory Integration for Free Road Space Detection"

# InternImage-XL + UperNet with Human-like Memory for Road Segmentation

A PyTorch implementation of **InternImage-XL backbone** combined with **UperNet decoder** and an **enhanced human-like memory system** for real-time road segmentation.

## 📁 Project Structure

```
upload/
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── internimage.py         # InternImage-XL backbone
│   ├── upernet.py             # UperNet decoder
│   ├── memory_system.py       # Human-like memory bank
│   └── memory_augmented_model.py  # Complete model
├── data/                      # Dataset utilities
│   ├── __init__.py
│   └── dataset.py            # Dataset class and utilities
├── utils/                     # Training utilities
│   ├── __init__.py
│   └── training_utils.py     # Loss, metrics, and utilities
├── config.py                 # Configuration settings
├── train.py                  # Training script
├── inference.py              # Inference script
└── requirements.txt          # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdkhanstd/MemRoadNet
   cd upload
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained weights**:
   - Download InternImage-XL pretrained weights to `pretrained/upernet_internimage_xl_640_160k_ade20k.pth`
   - Alternatively, update `config.py` with your pretrained weights path

## 📊 Dataset Structure

Organize your dataset as follows:
```
datasets/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── masks/
│       ├── image1.png
│       └── image2.png
└── val/
    ├── images/
    └── masks/
```

## 🚀 Training

1. **Configure settings** in `config.py`:
   ```python
   # Model Configuration
   BATCH_SIZE = 4
   LEARNING_RATE = 1e-4
   EPOCHS = 100
   
   # Memory System
   MEMORY_SIZE = 1000
   MEMORY_WEIGHT = 0.3
   USE_MEMORY_IN_TRAINING = True
   ```

2. **Start training**:
   ```bash
   python train.py
   ```

3. **Monitor progress**:
   - Training logs with memory statistics
   - Validation samples saved to `val_samples_intern/`
   - Best model saved to `weights_simple/best_internimage_upernet_with_memory.pth`

## 🔍 Inference

1. **Prepare test images** in `test_images/` directory

2. **Run inference**:
   ```bash
   python inference.py
   ```

3. **Results saved to**:
   - `inference_results/masks/` - Segmentation masks
   - `inference_results/inference_summary.txt` - Performance summary

## 🧠 Memory System

The human-like memory system includes:

- **Episodic Memory**: Specific experiences with rich context
- **Semantic Memory**: Generalized knowledge patterns  
- **Working Memory**: Recent short-term context
- **Emotional Valence**: Success/failure emotions guide consolidation
- **Forgetting Mechanism**: Gradual decay with importance-based retention
- **Consolidation**: Sleep-like memory strengthening between epochs

## ⚙️ Configuration

Key configuration options in `config.py`:

```python
class Config:
    # Model Architecture
    CHANNELS = 192
    DEPTHS = [5, 5, 24, 5]
    GROUPS = [12, 24, 48, 96]
    
    # Training
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    
    # Memory System
    MEMORY_SIZE = 1000
    TOP_K_MEMORIES = 5
    MEMORY_WEIGHT = 0.3
    MEMORY_CONSOLIDATION_INTERVAL = 5
    
    # Paths
    DATASET_DIR = "datasets"
    WEIGHTS_PATH = "./weights_simple"
    PRETRAINED_PATH = "pretrained/upernet_internimage_xl_640_160k_ade20k.pth"
```


## 📝 Citation

```bibtex

```


## 🙏 Acknowledgments

- InternImage team for the excellent backbone architecture
- UperNet authors for the segmentation head design
- PyTorch team for the deep learning framework
