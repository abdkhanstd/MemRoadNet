"""
Utility functions for training and evaluation
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import random
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def combined_loss(pred, target, bce_weight=0.4, dice_weight=0.6):
    """Enhanced combined BCE + Dice loss with better weighting"""
    # BCE loss
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    
    # Dice loss - using sigmoid on predictions
    pred_sigmoid = torch.sigmoid(pred)
    smooth = 1e-6  # Increased smoothing for numerical stability
    
    # Flatten tensors for dice calculation
    pred_flat = pred_sigmoid.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice
    
    # Weighted combination - emphasize dice loss more for sharp boundaries
    return bce_weight * bce + dice_weight * dice_loss

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU metric with improved thresholding"""
    # Apply sigmoid and threshold
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Flatten for calculation
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """Calculate pixel accuracy"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    correct_pixels = (pred_binary == target.float()).sum().item()
    total_pixels = target.numel()
    return correct_pixels / total_pixels

def save_validation_samples(images, preds, masks, epoch, save_dir="val_samples_intern", num_samples=5, test_images_dir=None):
    """Save validation samples and inference on test images"""
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        if len(images) == 0 or len(preds) == 0 or len(masks) == 0:
            logger.warning(f"Empty tensors for epoch {epoch}")
            return
            
        # Convert to numpy
        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
        else:
            images_np = images
            
        if isinstance(preds, torch.Tensor):
            # Apply sigmoid and sharper thresholding for clearer masks
            preds_sigmoid = torch.sigmoid(preds).detach().cpu().numpy()
            # Create binary mask with 0.5 threshold
            preds_binary = (preds_sigmoid > 0.5).astype(np.float32)
        else:
            preds_sigmoid = preds
            preds_binary = (preds_sigmoid > 0.5).astype(np.float32)
            
        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = masks
        
        # Sample indices
        num_samples = min(num_samples, len(images_np))
        indices = random.sample(range(len(images_np)), num_samples)
        
        for i, idx in enumerate(indices):
            try:
                # Get single sample
                image = images_np[idx]
                pred_prob = preds_sigmoid[idx]  # Probability map
                pred_binary = preds_binary[idx]  # Binary mask
                mask = masks_np[idx]
                
                # Handle tensor shapes
                if len(image.shape) == 3:  # [C, H, W]
                    image = image.transpose(1, 2, 0)  # [H, W, C]
                elif len(image.shape) == 4:  # [1, C, H, W]
                    image = image[0].transpose(1, 2, 0)
                    
                if len(pred_prob.shape) == 3:  # [1, H, W]
                    pred_prob = pred_prob[0]
                    pred_binary = pred_binary[0]
                elif len(pred_prob.shape) == 4:  # [1, 1, H, W]
                    pred_prob = pred_prob[0, 0]
                    pred_binary = pred_binary[0, 0]
                    
                if len(mask.shape) == 3:  # [1, H, W]
                    mask = mask[0]
                elif len(mask.shape) == 4:  # [1, 1, H, W]
                    mask = mask[0, 0]
                
                # Normalize image to [0, 255]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                # Convert masks to [0, 255] for visualization
                pred_prob_vis = (pred_prob * 255).astype(np.uint8)
                pred_binary_vis = (pred_binary * 255).astype(np.uint8)
                mask_vis = (mask * 255).astype(np.uint8)
                
                # Convert to 3-channel
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                
                pred_prob_3ch = np.stack([pred_prob_vis] * 3, axis=-1)
                pred_binary_3ch = np.stack([pred_binary_vis] * 3, axis=-1)
                mask_3ch = np.stack([mask_vis] * 3, axis=-1)
                
                # Create combined image [Original | Ground Truth | Probability | Binary Prediction]
                combined_image = np.concatenate((image, mask_3ch, pred_prob_3ch, pred_binary_3ch), axis=1)
                
                # Save
                filename = f"epoch_{epoch}_sample_{i}_combined.png"
                filepath = os.path.join(save_dir, filename)
                
                combined_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(filepath, combined_bgr)
                
                if not success:
                    logger.warning(f"Failed to save sample {i} for epoch {epoch}")
                    
            except Exception as e:
                logger.warning(f"Error saving sample {i} for epoch {epoch}: {e}")
                continue
                
        logger.info(f"✅ Saved {len(indices)} validation samples for epoch {epoch} in {save_dir}")
        
    except Exception as e:
        logger.error(f"Critical error in save_validation_samples for epoch {epoch}: {e}")

def load_model_with_memory(model_path, device):
    """Load model and restore memory state"""
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None, None
    
    try:
        logger.info(f"📄 Loading model and memory from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        return checkpoint
        
    except Exception as e:
        logger.error(f"⚠ Error loading model: {e}")
        return None, None

def save_model_with_memory(model, optimizer, epoch, best_iou, best_accuracy, save_path):
    """Save model with memory state - ensure patterns are on CPU"""
    memory_patterns_cpu = []
    for pattern in model.memory_bank.patterns:
        if hasattr(pattern, 'cpu'):
            memory_patterns_cpu.append(pattern.cpu())
        else:
            memory_patterns_cpu.append(pattern)
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'best_accuracy': best_accuracy,
        'memory_patterns': memory_patterns_cpu,
        'memory_contexts': model.memory_bank.contexts,
        'memory_emotions': model.memory_bank.emotions,
        'memory_timestamps': model.memory_bank.timestamps,
        'memory_importance_scores': model.memory_bank.importance_scores,
        'memory_access_counts': model.memory_bank.access_counts
    }, save_path)
    
    return True
