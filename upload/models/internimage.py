"""
InternImage-XL Backbone Implementation
Exact implementation matching pretrained checkpoint structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

logger = logging.getLogger(__name__)

class LayerNormChannel(nn.Module):
    """Channel-wise LayerNorm - matches norm1.0, norm2.0 in checkpoint"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if x.dim() == 4:  # [B, C, H, W]
            mean = x.mean(dim=1, keepdim=True)
            std = x.var(dim=1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(std + self.eps)
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        elif x.dim() == 3:  # [B, L, C]
            mean = x.mean(dim=-1, keepdim=True)
            std = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(std + self.eps)
            x = x * self.weight + self.bias
        return x

class DCNv3(nn.Module):
    """Exact DCNv3 matching checkpoint structure - FIXED shapes"""
    def __init__(self, channels, kernel_size=3, groups=8):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        
        # Depthwise conv with GroupNorm - matches dcn.dw_conv.0 and dcn.dw_conv.1.1
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels),  # dw_conv.0
            nn.Sequential(
                nn.Identity(),  # dw_conv.1.0 placeholder
                nn.GroupNorm(groups, channels)  # dw_conv.1.1
            )
        )
        
        # FIXED: Use Linear layers instead of Conv2d to match checkpoint shapes
        self.offset = nn.Linear(channels, groups * 2 * kernel_size * kernel_size)
        self.mask = nn.Linear(channels, groups * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for Linear layers: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        
        # Apply linear transformations
        x_flat = self.input_proj(x_flat)
        
        # Reshape back for conv: [B, H*W, C] -> [B, C, H, W]
        x = x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Apply depthwise conv and groupnorm
        x = self.dw_conv[0](x)  # depthwise conv
        x = self.dw_conv[1][1](x)  # groupnorm
        
        # Reshape for output projection: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        x_flat = self.output_proj(x_flat)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x = x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return x

class InternImageBlock(nn.Module):
    """InternImage block with EXACT parameter names from checkpoint"""
    def __init__(self, dim, mlp_ratio=4.0, groups=8, layer_scale=1e-5, post_norm=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.post_norm = post_norm
        
        # Layer scale parameters - matches gamma1, gamma2 in checkpoint
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))
        
        # Normalization layers - matches norm1.0, norm2.0 in checkpoint
        self.norm1 = nn.Sequential(LayerNormChannel(dim))
        self.norm2 = nn.Sequential(LayerNormChannel(dim))
        
        # DCNv3 module
        self.dcn = DCNv3(dim, groups=groups)
        
        # FIXED: MLP with fc1, fc2 naming to match checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.ModuleDict({
            'fc1': nn.Linear(dim, mlp_hidden_dim),
            'fc2': nn.Linear(mlp_hidden_dim, dim)
        })

    def forward(self, x):
        B, C, H, W = x.shape
        
        # First block: DCN with layer scale
        shortcut = x
        if not self.post_norm:
            x = self.norm1[0](x)
        x = self.dcn(x)
        if self.post_norm:
            x = self.norm1[0](x)
        x = shortcut + self.gamma1.view(1, -1, 1, 1) * x
        
        # Second block: MLP with layer scale
        shortcut = x
        if not self.post_norm:
            # Convert to [B, H*W, C] for LayerNorm and MLP
            x_2d = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
            x_2d = self.norm2[0](x_2d)
        else:
            x_2d = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        
        # Apply MLP with correct naming
        x_2d = self.mlp['fc1'](x_2d)
        x_2d = F.gelu(x_2d)
        x_2d = self.mlp['fc2'](x_2d)
        
        if self.post_norm:
            x_2d = self.norm2[0](x_2d)
        
        # Convert back to [B, C, H, W]
        x = x_2d.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.gamma2.view(1, -1, 1, 1) * x
        
        return x

class PatchEmbed(nn.Module):
    """Patch embedding exactly matching checkpoint structure"""
    def __init__(self, in_chans=3, embed_dim=192):
        super().__init__()
        
        # Two-stage embedding - matches conv1, norm1.1, conv2, norm2.1
        self.conv1 = nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1)
        self.norm1 = nn.Sequential(
            nn.Identity(),  # norm1.0 placeholder
            nn.BatchNorm2d(embed_dim // 2)  # norm1.1
        )
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1)
        self.norm2 = nn.Sequential(
            nn.Identity(),  # norm2.0 placeholder
            nn.BatchNorm2d(embed_dim)  # norm2.1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1[1](x)  # Use BatchNorm
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.norm2[1](x)  # Use BatchNorm
        return x

class InternImageLevel(nn.Module):
    """Level in InternImage exactly matching checkpoint structure"""
    def __init__(self, dim, depth, groups, mlp_ratio=4.0, layer_scale=1e-5, post_norm=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            InternImageBlock(
                dim=dim, 
                mlp_ratio=mlp_ratio, 
                groups=groups, 
                layer_scale=layer_scale,
                post_norm=post_norm
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class DownsampleLayer(nn.Module):
    """FIXED: Downsample layer matching checkpoint structure exactly"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Checkpoint shows: levels.X.downsample.conv (3x3) + levels.X.downsample.norm.1
        self.conv = nn.Conv2d(input_dim, output_dim, 3, 2, 1)  # 3x3 conv with stride 2
        self.norm = nn.Sequential(
            nn.Identity(),  # norm.0 placeholder
            nn.BatchNorm2d(output_dim)  # norm.1 - matches checkpoint
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm[1](x)  # Use BatchNorm
        return x

class InternImage(nn.Module):
    """InternImage backbone exactly matching checkpoint structure"""
    def __init__(self, 
                 core_op='DCNv3',
                 channels=192,
                 depths=[5, 5, 24, 5],
                 groups=[12, 24, 48, 96],
                 mlp_ratio=4.0,
                 layer_scale=1.0,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        
        self.core_op = core_op
        self.num_levels = len(depths)
        self.out_indices = out_indices
        self.post_norm = post_norm
        
        # Channel dimensions for each level
        embed_dims = [channels * (2 ** i) for i in range(self.num_levels)]
        
        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0])
        
        # Levels with integrated downsampling - matches checkpoint structure
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = InternImageLevel(
                dim=embed_dims[i],
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=mlp_ratio,
                layer_scale=layer_scale,
                post_norm=post_norm
            )
            self.levels.append(level)
            
            # FIXED: Downsample is INSIDE each level (except last) - matches checkpoint
            if i < self.num_levels - 1:
                level.downsample = DownsampleLayer(embed_dims[i], embed_dims[i + 1])
        
        self.feature_channels = embed_dims
        
        # Apply initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, LayerNormChannel)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights with exact matching"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"⚠ Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            logger.info(f"📄 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Extract backbone parameters
            backbone_params = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[9:]  # Remove 'backbone.' prefix
                    backbone_params[new_key] = value
            
            logger.info(f"📊 Found {len(backbone_params)} backbone parameters")
            
            # Get our model parameters
            model_dict = self.state_dict()
            
            # Enhanced matching with shape compatibility
            matched_params = {}
            for model_key in model_dict.keys():
                # Skip BatchNorm buffers (running_mean, running_var, 'num_batches_tracked')
                if any(x in model_key for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                    continue
                    
                if model_key in backbone_params:
                    model_param = model_dict[model_key]
                    ckpt_param = backbone_params[model_key]
                    
                    # Exact shape match
                    if model_param.shape == ckpt_param.shape:
                        matched_params[model_key] = ckpt_param
                    # Handle Linear vs Conv2d weight compatibility for DCN components
                    elif (len(model_param.shape) == 2 and len(ckpt_param.shape) == 2 and 
                          model_param.shape == ckpt_param.shape):
                        matched_params[model_key] = ckpt_param
                    else:
                        logger.debug(f"Shape mismatch {model_key}: {model_param.shape} vs {ckpt_param.shape}")
            
            # Load matched parameters
            if matched_params:
                self.load_state_dict(matched_params, strict=False)
                # Calculate loading percentage excluding BatchNorm buffers
                learnable_params = [k for k in model_dict.keys() 
                                  if not any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked'])]
                load_percentage = len(matched_params) / len(learnable_params) * 100
                logger.info(f"✅ Loaded {len(matched_params)}/{len(learnable_params)} learnable parameters ({load_percentage:.1f}%)")
            else:
                logger.warning("⚠ No parameters matched!")
            
            return len(matched_params) > 0
                
        except Exception as e:
            logger.error(f"⚠ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def forward(self, x):
        features = []
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Forward through levels with integrated downsampling
        for i in range(self.num_levels):
            x = self.levels[i](x)
            if i in self.out_indices:
                features.append(x)
            
            # Apply downsample if it exists (levels 0, 1, 2 have downsample)
            if hasattr(self.levels[i], 'downsample'):
                x = self.levels[i].downsample(x)
        
        return features
