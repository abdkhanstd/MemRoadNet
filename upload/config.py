"""
Configuration for InternImage-XL + UperNet with Human-like Memory
Road Segmentation Model
"""
import os

class Config:
    # Model Architecture
    CHANNELS = 192
    DEPTHS = [5, 5, 24, 5]
    GROUPS = [12, 24, 48, 96]
    MLP_RATIO = 4.0
    LAYER_SCALE = 1.0
    OFFSET_SCALE = 2.0
    POST_NORM = True
    NUM_CLASSES = 1  # Binary road segmentation
    
    # Training Configuration
    IMAGE_SIZE = (640, 640)
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    # Memory System Configuration
    MEMORY_SIZE = 1000
    TOP_K_MEMORIES = 5
    MEMORY_WEIGHT = 0.3
    USE_MEMORY_IN_TRAINING = True
    USE_MEMORY_IN_INFERENCE = True
    MEMORY_CONSOLIDATION_INTERVAL = 5
    
    # Loss Function
    BCE_WEIGHT = 0.4
    DICE_WEIGHT = 0.6
    
    # Data Loading
    num_workers = 4
    prefetch_factor = 2
    
    # Paths
    WEIGHTS_PATH = "./weights_simple"
    PRETRAINED_PATH = "pretrained/upernet_internimage_xl_640_160k_ade20k.pth"
    DATASET_DIR = "datasets"
    TEST_IMAGES_DIR = "./test_images"
    OUTPUT_DIR = "./inference_results"
    
    # Inference Configuration
    THRESHOLD = 0.5
    MODEL_PATH = "./weights_simple/best_internimage_upernet_with_memory.pth"
    
    # Validation
    SAVE_VALIDATION_SAMPLES = True
    VAL_SAMPLES_DIR = "val_samples_intern"
    
    def __init__(self):
        # Create necessary directories
        os.makedirs(self.WEIGHTS_PATH, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if self.SAVE_VALIDATION_SAMPLES:
            os.makedirs(self.VAL_SAMPLES_DIR, exist_ok=True)

# Global config instance
config = Config()
