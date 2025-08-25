#!/usr/bin/env python3
"""
Training Script for InternImage-XL + UperNet with Human-like Memory
Road Segmentation Model
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import random
import numpy as np

# Select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Local imports
from config import config
from models import MemoryAugmentedInternImage
from data import RoadDataset, custom_collate_fn, get_dataset_samples, get_transforms
from utils import (
    combined_loss, 
    calculate_iou, 
    calculate_pixel_accuracy, 
    save_validation_samples,
    save_model_with_memory
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")

def train_model_with_memory(model, train_loader, val_loader):
    """Enhanced training loop with human-like memory integration"""
    logger.info("🚀 Starting InternImage-XL + UperNet Training with Enhanced Memory...")
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = GradScaler('cuda')
    
    best_iou = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        model.set_epoch(epoch)  # Update epoch for memory context
        model.train()
        
        epoch_loss = 0
        epoch_iou = 0
        epoch_accuracy = 0
        count = 0
        memory_additions = 0
        
        # Training with tqdm showing running metrics + memory stats
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        running_loss = 0
        running_iou = 0
        running_accuracy = 0
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            if images is None or masks is None:
                continue
                
            images, masks = images.to(device), masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                # Forward pass with memory integration
                pred_masks = model(images, masks, use_memory=config.USE_MEMORY_IN_TRAINING)
                loss = combined_loss(pred_masks, masks.float(), config.BCE_WEIGHT, config.DICE_WEIGHT)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate metrics
            iou = calculate_iou(pred_masks, masks)
            accuracy = calculate_pixel_accuracy(pred_masks, masks)
            
            # Add experience to enhanced memory (both successes and failures)
            model.add_to_memory(images, masks, iou, loss)
            memory_additions += 1
            
            # Update running metrics
            running_loss = (running_loss * count + loss.item()) / (count + 1)
            running_iou = (running_iou * count + iou) / (count + 1)
            running_accuracy = (running_accuracy * count + accuracy) / (count + 1)
            
            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_accuracy += accuracy
            count += 1
            
            # Update tqdm description with running metrics + memory info
            memory_stats = f"Mem:{len(model.memory_bank.patterns)}"
            if len(model.memory_bank.emotions) > 0:
                positive_memories = sum(1 for e in model.memory_bank.emotions 
                                      if e in ['positive', 'very_positive'])
                memory_stats += f"|Pos:{positive_memories}"
            
            train_pbar.set_postfix({
                'Loss': f'{running_loss:.4f}',
                'IoU': f'{running_iou:.4f}',
                'Acc': f'{running_accuracy:.4f}',
                'Memory': memory_stats
            })
        
        scheduler.step()
        
        # Memory consolidation between epochs
        if epoch % config.MEMORY_CONSOLIDATION_INTERVAL == 0:
            logger.info(f"🧠 Performing memory consolidation at epoch {epoch+1}")
            model.consolidate_memory()
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_accuracy = 0
        val_count = 0
        validation_predictions = None
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            running_val_loss = 0
            running_val_iou = 0
            running_val_accuracy = 0
            
            for val_batch_idx, (images, masks) in enumerate(val_pbar):
                if images is None or masks is None:
                    continue
                    
                images, masks = images.to(device), masks.to(device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                
                with autocast(device_type='cuda'):
                    # Use memory during validation too
                    pred_masks = model(images, masks, use_memory=True)
                    loss = combined_loss(pred_masks, masks.float(), config.BCE_WEIGHT, config.DICE_WEIGHT)
                
                # Calculate metrics
                iou = calculate_iou(pred_masks, masks)
                accuracy = calculate_pixel_accuracy(pred_masks, masks)
                
                # Update running validation metrics
                running_val_loss = (running_val_loss * val_count + loss.item()) / (val_count + 1)
                running_val_iou = (running_val_iou * val_count + iou) / (val_count + 1)
                running_val_accuracy = (running_val_accuracy * val_count + accuracy) / (val_count + 1)
                
                val_loss += loss.item()
                val_iou += iou
                val_accuracy += accuracy
                val_count += 1
                
                # Update validation tqdm
                val_pbar.set_postfix({
                    'Val_Loss': f'{running_val_loss:.4f}',
                    'Val_IoU': f'{running_val_iou:.4f}',
                    'Val_Acc': f'{running_val_accuracy:.4f}'
                })
                
                # Save first batch for visualization
                if val_batch_idx == 0:
                    validation_predictions = (images, pred_masks, masks)
        
        # Logging with memory statistics
        if count > 0 and val_count > 0:
            avg_loss = epoch_loss / count
            avg_iou = epoch_iou / count
            avg_accuracy = epoch_accuracy / count
            avg_val_loss = val_loss / val_count
            avg_val_iou = val_iou / val_count
            avg_val_accuracy = val_accuracy / val_count
            current_lr = optimizer.param_groups[0]['lr']
            memory_size = len(model.memory_bank.patterns)
            working_memory_size = len(model.memory_bank.working_memory)
            
            # Memory emotion distribution
            emotion_dist = {}
            for emotion in model.memory_bank.emotions:
                emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
            
            logger.info(
                f"Epoch {epoch+1}/{config.EPOCHS}: "
                f"Loss={avg_loss:.4f}, IoU={avg_iou:.4f}, Acc={avg_accuracy:.4f}, "
                f"Val_Loss={avg_val_loss:.4f}, Val_IoU={avg_val_iou:.4f}, Val_Acc={avg_val_accuracy:.4f}, "
                f"LR={current_lr:.6f}"
            )
            logger.info(
                f"🧠 Memory: {memory_size} long-term, {working_memory_size} working, "
                f"Emotions: {emotion_dist}"
            )
            
            # Save validation samples
            if validation_predictions is not None and config.SAVE_VALIDATION_SAMPLES:
                val_images, val_preds, val_masks = validation_predictions
                save_validation_samples(
                    val_images.cpu(), 
                    val_preds.cpu(), 
                    val_masks.cpu(), 
                    epoch + 1,
                    save_dir=config.VAL_SAMPLES_DIR,
                    num_samples=3,
                    test_images_dir=config.TEST_IMAGES_DIR
                )
            
            # Save best model
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                patience_counter = 0
                
                # Save model with memory state
                save_path = os.path.join(config.WEIGHTS_PATH, "best_internimage_upernet_with_memory.pth")
                save_model_with_memory(model, optimizer, epoch, best_iou, avg_val_accuracy, save_path)
                
                logger.info(f"✅ New best model saved with IoU: {best_iou:.4f}, Accuracy: {avg_val_accuracy:.4f}")
                logger.info(f"🧠 Memory state saved: {memory_size} memories with rich context")
                
                # Save best model validation samples
                if validation_predictions is not None and config.SAVE_VALIDATION_SAMPLES:
                    val_images, val_preds, val_masks = validation_predictions
                    save_validation_samples(
                        val_images.cpu(), 
                        val_preds.cpu(), 
                        val_masks.cpu(), 
                        f"BEST_epoch_{epoch+1}",
                        save_dir=config.VAL_SAMPLES_DIR,
                        num_samples=5
                    )
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best IoU: {best_iou:.4f}")
                break
        
        torch.cuda.empty_cache()
    
    logger.info(f"✅ Training completed! Best IoU: {best_iou:.4f}")
    logger.info(f"🧠 Final memory state: {len(model.memory_bank.patterns)} long-term memories")
    return best_iou

def main():
    """Enhanced main training pipeline with memory integration"""
    logger.info("🎯 InternImage-XL + UperNet Road Segmentation with Enhanced Human-like Memory")
    logger.info(f"Weights: {config.WEIGHTS_PATH}")
    logger.info(f"Pretrained: {config.PRETRAINED_PATH}")
    logger.info(f"Device: {device}")
    logger.info(f"Validation samples will be saved to: {config.VAL_SAMPLES_DIR}/")
    logger.info(f"Memory configuration:")
    logger.info(f"Max memory size: {config.MEMORY_SIZE}")
    logger.info(f"Memory weight: {config.MEMORY_WEIGHT}")
    logger.info(f"Use memory in training: {config.USE_MEMORY_IN_TRAINING}")
    logger.info(f"Memory consolidation interval: {config.MEMORY_CONSOLIDATION_INTERVAL} epochs")
    
    # Use datasets directory
    logger.info(f"🔍 Scanning dataset directory: {config.DATASET_DIR}")
    all_samples = get_dataset_samples(config.DATASET_DIR)
    logger.info(f"📊 Total samples found: {len(all_samples)}")
    if len(all_samples) == 0:
        logger.error(f"No samples found in {config.DATASET_DIR}")
        return
    train_samples, val_samples = train_test_split(all_samples, test_size=config.VALIDATION_SPLIT, random_state=42)
    logger.info(f"📊 Training samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    # Define transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = RoadDataset(train_samples, transform=train_transform)
    val_dataset = RoadDataset(val_samples, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    logger.info(f"📊 Final dataset summary:")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation/Test samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error("⚠ No training samples found! Check dataset directory structure.")
        return
    if len(val_dataset) == 0:
        logger.error("⚠ No validation samples found! Check dataset directory structure.")
        return

    # Create model
    model = MemoryAugmentedInternImage(
        memory_weight=config.MEMORY_WEIGHT, 
        pretrained_path=config.PRETRAINED_PATH
    ).to(device)

    # Train with enhanced memory
    best_iou = train_model_with_memory(model, train_loader, val_loader)

    # Final evaluation with memory analysis
    logger.info("🎯 Running final evaluation with memory analysis...")
    model.eval()
    final_iou = 0
    sample_count = 0
    memory_usage_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Final Evaluation")):
            if images is None or masks is None:
                continue
                
            images, masks = images.to(device), masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            # Test both with and without memory
            pred_masks_no_memory = model(images, masks, use_memory=False)
            pred_masks_with_memory = model(images, masks, use_memory=True)
            
            iou_no_memory = calculate_iou(pred_masks_no_memory, masks)
            iou_with_memory = calculate_iou(pred_masks_with_memory, masks)
            
            final_iou += iou_with_memory
            sample_count += 1
            
            if iou_with_memory > iou_no_memory:
                memory_usage_count += 1
            
            # Save final samples from first few batches
            if batch_idx < 3 and config.SAVE_VALIDATION_SAMPLES:
                save_validation_samples(
                    images.cpu(), 
                    pred_masks_with_memory.cpu(), 
                    masks.cpu(), 
                    f"FINAL_batch_{batch_idx}_with_memory",
                    save_dir=config.VAL_SAMPLES_DIR,
                    num_samples=2
                )
                
                save_validation_samples(
                    images.cpu(), 
                    pred_masks_no_memory.cpu(), 
                    masks.cpu(), 
                    f"FINAL_batch_{batch_idx}_no_memory",
                    save_dir=config.VAL_SAMPLES_DIR,
                    num_samples=2
                )
    
    if sample_count > 0:
        final_avg_iou = final_iou / sample_count
        memory_improvement_rate = memory_usage_count / sample_count * 100
        logger.info(f"📊 Final Evaluation IoU (with memory): {final_avg_iou:.4f}")
        logger.info(f"🧠 Memory improved predictions in {memory_improvement_rate:.1f}% of cases")
    
    # Memory analysis
    logger.info(f"Final Memory Analysis:")
    logger.info(f"Total long-term memories: {len(model.memory_bank.patterns)}")
    logger.info(f"Working memory size: {len(model.memory_bank.working_memory)}")
    
    if model.memory_bank.emotions:
        emotion_counts = {}
        for emotion in model.memory_bank.emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        logger.info(f"Emotion distribution: {emotion_counts}")
        
        # Most accessed memories
        if model.memory_bank.access_counts:
            max_access = max(model.memory_bank.access_counts)
            most_accessed_idx = model.memory_bank.access_counts.index(max_access)
            logger.info(f"Most accessed memory: {max_access} times, emotion: {model.memory_bank.emotions[most_accessed_idx]}")
        
        # Average importance scores
        if model.memory_bank.importance_scores: 
            avg_importance = sum(model.memory_bank.importance_scores) / len(model.memory_bank.importance_scores)
            logger.info(f"Average memory importance: {avg_importance:.4f}")
    
    logger.info(f"Training Summary:")
    logger.info(f"Best Validation IoU: {best_iou:.4f}")
    logger.info(f"Final Evaluation IoU: {final_avg_iou:.4f}")
    logger.info(f"Memory improvement rate: {memory_improvement_rate:.1f}%")
    logger.info(f"Model saved: {config.WEIGHTS_PATH}/best_internimage_upernet_with_memory.pth")
    logger.info(f"Validation samples: {config.VAL_SAMPLES_DIR}/")
    logger.info(f"Training and evaluation completed successfully with enhanced human-like memory!")

if __name__ == "__main__":
    main()
