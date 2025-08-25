"""
Memory-Augmented InternImage Model
Combines InternImage-XL backbone + UperNet decoder + Human-like Memory System
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from .internimage import InternImage
from .upernet import UperNetDecodeHead
from .memory_system import HumanLikeMemoryBank

logger = logging.getLogger(__name__)

class MemoryAugmentedInternImage(nn.Module):
    """InternImage-XL + UperNet with Enhanced Human-like Memory Integration"""
    
    def __init__(self, memory_weight=0.3, pretrained_path=None):
        super().__init__()
        
        # InternImage-XL backbone with exact config
        self.backbone = InternImage(
            core_op='DCNv3',
            channels=192,
            depths=[5, 5, 24, 5],
            groups=[12, 24, 48, 96],
            mlp_ratio=4.0,
            layer_scale=1.0,
            offset_scale=2.0,
            post_norm=True,
            with_cp=False,
            out_indices=(0, 1, 2, 3)
        )
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            success = self.backbone.load_pretrained_weights(pretrained_path)
            if success:
                logger.info("🎯 Successfully loaded pretrained backbone weights!")
            else:
                logger.warning("⚠️ Failed to load pretrained weights")
        elif pretrained_path:
            logger.warning(f"⚠ Pretrained weights not found: {pretrained_path}")
        
        # UperNet decode head exactly matching checkpoint
        self.decode_head = UperNetDecodeHead(
            in_channels=self.backbone.feature_channels,  # [192, 384, 768, 1536]
            num_classes=1,  # For road segmentation
            channels=512
        )
        
        # Enhanced Memory system
        self.memory_bank = HumanLikeMemoryBank(
            max_size=1000, 
            feature_dim=128
        )
        
        # Memory integration components
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_compressor = nn.Linear(self.backbone.feature_channels[-1], 128)
        self.memory_weight = memory_weight
        
        # Memory attention and fusion layers
        self.memory_attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.memory_fusion = nn.Linear(256, 128)
        
        # Memory influence on segmentation
        self.memory_influence = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1536)  # Match deepest feature channel
        )
        
        # Store query features and context for memory operations
        self.query_features = None
        self.current_context = None
        self.current_epoch = 0
        
    def extract_context(self, x, pred_masks=None, loss=None):
        """Extract rich contextual information from current situation"""
        context = {
            'image_stats': {
                'mean_brightness': x.mean().item(),
                'std_brightness': x.std().item(),
                'contrast': (x.max() - x.min()).item(),
                'channel_means': [x[:, i].mean().item() for i in range(x.size(1))]
            },
            'spatial_info': {
                'height': x.size(2),
                'width': x.size(3),
                'aspect_ratio': x.size(3) / x.size(2)
            },
            'training_context': {
                'epoch': self.current_epoch,
                'batch_size': x.size(0),
                'is_training': self.training
            }
        }
        
        if pred_masks is not None:
            context['prediction_stats'] = {
                'road_ratio': torch.sigmoid(pred_masks).mean().item(),
                'confidence': torch.sigmoid(pred_masks).std().item()
            }
        
        if loss is not None:
            context['loss_value'] = loss.item()
        
        return context
    
    def forward(self, x, masks=None, use_memory=True):
        input_size = x.shape[-2:]
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Extract query for memory operations
        query_vector = self.global_pool(features[-1]).flatten(1)  # [B, 1536]
        self.query_features = self.feature_compressor(query_vector)  # [B, 128]
        
        # Memory-augmented features
        memory_enhanced_features = features[-1]
        
        if use_memory and len(self.memory_bank.patterns) > 0:
            # Recall relevant memories
            recalled_memories = self.memory_bank.recall(
                self.query_features[0],  # Use first sample in batch
                query_context=self.current_context,
                top_k=5,
                recall_type='associative'
            )
            
            if recalled_memories:
                # Convert memories to tensor and move to correct device
                memory_patterns = torch.stack([
                    m['pattern'] for m in recalled_memories
                ]).to(x.device)  # [K, 128]
                
                # Apply attention between query and memories
                batch_size = self.query_features.size(0)
                memory_patterns = memory_patterns.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [B, K, 128]
                
                query_expanded = self.query_features.unsqueeze(1)  # [B, 1, 128]
                
                attended_memory, attention_weights = self.memory_attention(
                    query_expanded, memory_patterns, memory_patterns
                )  # [B, 1, 128]
                
                # Fuse current features with memory
                combined_features = torch.cat([
                    self.query_features, 
                    attended_memory.squeeze(1)
                ], dim=-1)  # [B, 256]
                
                memory_enhanced_query = self.memory_fusion(combined_features)  # [B, 128]
                
                # Generate memory influence on deep features
                memory_influence = self.memory_influence(memory_enhanced_query)  # [B, 1536]
                
                # Apply memory influence to deepest features
                B, C, H, W = features[-1].shape
                memory_influence = memory_influence.view(B, C, 1, 1)
                memory_enhanced_features = features[-1] + self.memory_weight * memory_influence
                
                # Update features list with memory-enhanced version
                features = features[:-1] + [memory_enhanced_features]
                
                # Store enhanced query for memory update
                self.query_features = memory_enhanced_query
        
        # Segmentation with potentially memory-enhanced features
        seg_logits = self.decode_head(features)
        
        # Resize to input size
        seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)
        
        return seg_logits
    
    def add_to_memory(self, image, mask, iou_score, loss_value=None):
        """Add experience to enhanced memory system"""
        if self.query_features is not None:
            # Extract rich context
            context = self.extract_context(image, mask, loss_value)
            self.current_context = context
            
            # Add experience to memory (both successes and failures)
            self.memory_bank.add_experience(
                pattern=self.query_features[0] if self.query_features.dim() > 1 else self.query_features,
                context=context,
                success_score=iou_score,
                metadata={
                    'epoch': self.current_epoch,
                    'loss': loss_value.item() if loss_value is not None else None,
                    'batch_size': image.size(0)
                }
            )
    
    def consolidate_memory(self):
        """Perform memory consolidation (call between epochs)"""
        self.memory_bank.consolidate_during_sleep()
    
    def set_epoch(self, epoch):
        """Update current epoch for memory context"""
        self.current_epoch = epoch
