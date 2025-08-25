"""
Dataset utilities for road segmentation
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

class RoadDataset(Dataset):
    """Road segmentation dataset using directory structure, not CSV"""
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of (image_path, mask_path)
        self.transform = transform
        logger.info(f"📊 Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                return None, None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                return None, None
            mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask)
            return image, mask
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            return None, None

def custom_collate_fn(batch):
    """Handle None values in batch"""
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

def get_dataset_samples(dataset_dir):
    """Scan dataset_dir for all image-mask pairs"""
    samples = []
    for subdir in glob.glob(os.path.join(dataset_dir, '*')):
        images_dir = os.path.join(subdir, 'images')
        masks_dir = os.path.join(subdir, 'masks')
        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            continue
        image_files = sorted(glob.glob(os.path.join(images_dir, '*')))
        mask_files = sorted(glob.glob(os.path.join(masks_dir, '*')))
        # Match by filename (without extension)
        image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        mask_map = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}
        for key in image_map:
            if key in mask_map:
                samples.append((image_map[key], mask_map[key]))
    return samples

def get_transforms(train=True):
    """Get data transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])
