#!/usr/bin/env python3
"""
Inference Script for InternImage-XL + UperNet with Enhanced Memory
Loads saved model and memory state, performs inference on test images
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging
import time

# Select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Local imports
from config import config
from models import MemoryAugmentedInternImage
from data import RoadDataset, custom_collate_fn, get_transforms
from utils import calculate_iou, calculate_pixel_accuracy, load_model_with_memory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")

class InferenceDataset(torch.utils.data.Dataset):
    """Simple dataset for inference on individual images"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                return None, img_path
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            image = cv2.resize(image, config.IMAGE_SIZE)
            
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
            return image, img_path, original_size
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            return None, img_path, None

def inference_collate_fn(batch):
    """Handle None values in inference batch"""
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0:
        return None, None, None
    
    images, paths, sizes = zip(*valid_batch)
    images = torch.stack(images)
    return images, paths, sizes

def load_trained_model(model_path):
    """Load trained model with memory state"""
    checkpoint = load_model_with_memory(model_path, device)
    if checkpoint is None:
        return None
        
    try:
        # Create model
        model = MemoryAugmentedInternImage(
            memory_weight=config.MEMORY_WEIGHT, 
            pretrained_path=config.PRETRAINED_PATH
        ).to(device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore memory state
        if 'memory_patterns' in checkpoint and checkpoint['memory_patterns']:
            model.memory_bank.patterns = checkpoint['memory_patterns']
            model.memory_bank.contexts = checkpoint.get('memory_contexts', [])
            model.memory_bank.emotions = checkpoint.get('memory_emotions', [])
            model.memory_bank.timestamps = checkpoint.get('memory_timestamps', [])
            model.memory_bank.importance_scores = checkpoint.get('memory_importance_scores', [])
            model.memory_bank.access_counts = checkpoint.get('memory_access_counts', [])
            
            logger.info(f"Restored memory state: {len(model.memory_bank.patterns)} memories")
            
            # Log memory statistics
            emotion_counts = {}
            for emotion in model.memory_bank.emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            logger.info(f"Memory emotions: {emotion_counts}")
        else:
            logger.info("No memory state found in checkpoint")
        
        model.eval()
        return model, checkpoint
        
    except Exception as e:
        logger.error(f"⚠ Error loading model: {e}")
        return None, None

def run_inference_on_images(model, image_paths, use_memory=True, save_results=True):
    """Run inference on a list of images"""
    logger.info(f"🔍 Running inference on {len(image_paths)} images")
    logger.info(f"Memory usage: {'Enabled' if use_memory else 'Disabled'}")
    
    # Create dataset and dataloader
    inference_transform = get_transforms(train=False)
    inference_dataset = InferenceDataset(image_paths, transform=inference_transform)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=inference_collate_fn
    )
    
    results = []
    total_inference_time = 0
    
    with torch.no_grad():
        for batch_idx, (images, paths, original_sizes) in enumerate(tqdm(inference_loader, desc="Inference")):
            if images is None:
                continue
            
            images = images.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Run inference
            pred_masks = model(images, use_memory=use_memory)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Process predictions
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            pred_masks_binary = (pred_masks_sigmoid > config.THRESHOLD).float()
            
            # Move to CPU for processing
            pred_masks_sigmoid = pred_masks_sigmoid.cpu().numpy()
            pred_masks_binary = pred_masks_binary.cpu().numpy()
            
            # Save results for each image in batch
            for i, (path, original_size) in enumerate(zip(paths, original_sizes)):
                if original_size is None:
                    continue
                    
                # Get prediction for this image
                pred_prob = pred_masks_sigmoid[i, 0]  # [H, W]
                pred_binary = pred_masks_binary[i, 0]  # [H, W]
                
                # Resize back to original size
                pred_prob_resized = cv2.resize(pred_prob, (original_size[1], original_size[0]))
                pred_binary_resized = cv2.resize(pred_binary, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                
                # Save results
                if save_results:
                    save_single_result(path, pred_prob_resized, pred_binary_resized, original_size)
                
                results.append({
                    'image_path': path,
                    'pred_prob': pred_prob_resized,
                    'pred_binary': pred_binary_resized,
                    'original_size': original_size,
                    'inference_time': inference_time / len(paths)  # Average time per image
                })
    
    avg_inference_time = total_inference_time / len(image_paths) if len(image_paths) > 0 else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    logger.info(f"✅ Inference completed!")
    logger.info(f"📊 Average inference time: {avg_inference_time:.4f}s per image")
    logger.info(f"📊 FPS: {fps:.2f}")
    
    return results

def save_single_result(image_path, pred_prob, pred_binary, original_size):
    """Save inference result for a single image"""
    # Create output directory structure
    rel_path = os.path.relpath(image_path, config.TEST_IMAGES_DIR)
    output_dir = os.path.join(config.OUTPUT_DIR, "masks", os.path.dirname(rel_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save probability map
    prob_path = os.path.join(output_dir, f"{base_name}_prob.png")
    prob_vis = (pred_prob * 255).astype(np.uint8)
    cv2.imwrite(prob_path, prob_vis)
    
    # Save binary mask
    binary_path = os.path.join(output_dir, f"{base_name}_mask.png")
    binary_vis = (pred_binary * 255).astype(np.uint8)
    cv2.imwrite(binary_path, binary_vis)
    
    # Save combined visualization
    try:
        original_image = cv2.imread(image_path)
        if original_image is not None:
            # Resize to match prediction
            if original_image.shape[:2] != original_size:
                original_image = cv2.resize(original_image, (original_size[1], original_size[0]))
            
            # Create overlay
            overlay = original_image.copy()
            road_pixels = pred_binary > 0.5
            overlay[road_pixels] = [0, 255, 0]  # Green for roads
            
            # Blend with original
            combined = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
            
            combined_path = os.path.join(output_dir, f"{base_name}_combined.png")
            cv2.imwrite(combined_path, combined)
    except Exception as e:
        logger.warning(f"Failed to create combined visualization for {image_path}: {e}")

def evaluate_on_test_set(model, test_samples):
    """Evaluate model on test set with ground truth"""
    logger.info(f"📊 Evaluating on {len(test_samples)} test samples")
    
    test_dataset = RoadDataset(test_samples, transform=get_transforms(train=False))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    total_iou = 0
    total_accuracy = 0
    count = 0
    memory_improvements = 0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluation"):
            if images is None or masks is None:
                continue
                
            images, masks = images.to(device), masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            # Test with and without memory
            pred_no_memory = model(images, masks, use_memory=False)
            pred_with_memory = model(images, masks, use_memory=config.USE_MEMORY_IN_INFERENCE)
            
            iou_no_memory = calculate_iou(pred_no_memory, masks)
            iou_with_memory = calculate_iou(pred_with_memory, masks)
            accuracy = calculate_pixel_accuracy(pred_with_memory, masks)
            
            total_iou += iou_with_memory
            total_accuracy += accuracy
            count += 1
            
            if iou_with_memory > iou_no_memory:
                memory_improvements += 1
    
    if count > 0:
        avg_iou = total_iou / count
        avg_accuracy = total_accuracy / count
        memory_improvement_rate = memory_improvements / count * 100
        
        logger.info(f"📊 Evaluation Results:")
        logger.info(f"Average IoU: {avg_iou:.4f}")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Memory improved {memory_improvement_rate:.1f}% of predictions")
        
        return avg_iou, avg_accuracy, memory_improvement_rate
    
    return 0, 0, 0

def main():
    """Main inference pipeline"""
    logger.info("🔍 InternImage-XL + UperNet Inference with Enhanced Memory")
    logger.info(f"Model path: {config.MODEL_PATH}")
    logger.info(f"Test images: {config.TEST_IMAGES_DIR}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Memory usage in inference: {config.USE_MEMORY_IN_INFERENCE}")
    
    # Load trained model
    logger.info("📄 Loading trained model...")
    model, checkpoint = load_trained_model(config.MODEL_PATH)
    if model is None:
        logger.error("Failed to load model!")
        return
    
    logger.info(f"✅ Model loaded successfully!")
    if checkpoint:
        logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Best IoU: {checkpoint.get('best_iou', 'unknown'):.4f}")
    
    # Get test images
    if os.path.exists(config.TEST_IMAGES_DIR):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        test_image_paths = []
        for ext in image_extensions:
            test_image_paths.extend(glob.glob(os.path.join(config.TEST_IMAGES_DIR, '**', ext), recursive=True))
        
        logger.info(f"📊 Found {len(test_image_paths)} test images")
        
        if len(test_image_paths) > 0:
            # Run inference
            results = run_inference_on_images(
                model, 
                test_image_paths, 
                use_memory=config.USE_MEMORY_IN_INFERENCE,
                save_results=True
            )
            
            # Save inference summary
            summary_path = os.path.join(config.OUTPUT_DIR, "inference_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Inference Summary\\n")
                f.write(f"================\\n")
                f.write(f"Model: {config.MODEL_PATH}\\n")
                f.write(f"Total images: {len(test_image_paths)}\\n")
                f.write(f"Memory usage: {config.USE_MEMORY_IN_INFERENCE}\\n")
                f.write(f"Threshold: {config.THRESHOLD}\\n")
                
                if results:
                    avg_time = sum(r['inference_time'] for r in results) / len(results)
                    f.write(f"Average inference time: {avg_time:.4f}s\\n")
                    f.write(f"FPS: {1.0/avg_time:.2f}\\n")
                
                f.write(f"\\nResults saved to: {config.OUTPUT_DIR}/masks/\\n")
            
            logger.info(f"📄 Inference summary saved to: {summary_path}")
        else:
            logger.warning("No test images found!")
    else:
        logger.warning(f"Test images directory not found: {config.TEST_IMAGES_DIR}")
    
    # If we have validation data, run evaluation
    try:
        from data import get_dataset_samples
        from sklearn.model_selection import train_test_split
        
        if os.path.exists(config.DATASET_DIR):
            logger.info("📊 Running evaluation on validation set...")
            all_samples = get_dataset_samples(config.DATASET_DIR)
            if len(all_samples) > 0:
                _, val_samples = train_test_split(all_samples, test_size=config.VALIDATION_SPLIT, random_state=42)
                evaluate_on_test_set(model, val_samples)
    except Exception as e:
        logger.info(f"Evaluation skipped: {e}")
    
    logger.info("✅ Inference completed successfully!")

if __name__ == "__main__":
    main()
