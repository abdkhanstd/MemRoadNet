"""
UperNet Decoder Implementation
Exact implementation matching pretrained checkpoint structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class PSPModule(nn.Module):
    """PSP module exactly matching checkpoint structure"""
    def __init__(self, in_channels, out_channels, pool_scales):
        super().__init__()
        self.pool_scales = pool_scales
        
        # PSP modules - matches psp_modules.X.1 structure in checkpoint
        self.psp_modules = nn.ModuleList()
        for i, scale in enumerate(pool_scales):
            psp_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Sequential(
                    nn.Identity(),  # psp_modules.X.0 (AdaptiveAvgPool2d)
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),  # psp_modules.X.1.conv
                        nn.BatchNorm2d(out_channels),  # psp_modules.X.1.bn
                        nn.ReLU(True)
                    )
                )
            )
            self.psp_modules.append(psp_module)
        
        # Bottleneck - matches bottleneck structure in checkpoint
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, 3, padding=1, bias=False),  # bottleneck.conv
            nn.BatchNorm2d(out_channels),  # bottleneck.bn
            nn.ReLU(True)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        psp_outs = [x]
        
        for psp_module in self.psp_modules:
            psp_out = psp_module[0](x)  # AdaptiveAvgPool2d
            psp_out = psp_module[1][1](psp_out)  # conv + bn + relu
            psp_out = F.interpolate(psp_out, size=input_size, mode='bilinear', align_corners=False)
            psp_outs.append(psp_out)
            
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

class UperNetDecodeHead(nn.Module):
    """UperNet decode head exactly matching checkpoint structure"""
    def __init__(self, in_channels, num_classes=1, channels=512, pool_scales=[1, 2, 3, 6]):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        
        # Lateral convolutions - matches lateral_convs.X structure
        self.lateral_convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_ch, channels, 1, bias=False),  # lateral_convs.X.conv
                nn.BatchNorm2d(channels),  # lateral_convs.X.bn
                nn.ReLU(True)
            )
            self.lateral_convs.append(lateral_conv)
        
        # FPN convolutions - matches fpn_convs.X structure
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            fpn_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),  # fpn_convs.X.conv
                nn.BatchNorm2d(channels),  # fpn_convs.X.bn
                nn.ReLU(True)
            )
            self.fpn_convs.append(fpn_conv)
        
        # PSP module for highest level - matches psp_modules and bottleneck
        self.psp_modules = PSPModule(
            in_channels=in_channels[-1],
            out_channels=channels,
            pool_scales=pool_scales
        )
        
        # Final classifier - matches conv_seg in checkpoint
        self.conv_seg = nn.Conv2d(len(in_channels) * channels, num_classes, 1)

    def forward(self, features):
        # Build lateral connections
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i == len(self.lateral_convs) - 1:
                # Apply PSP to deepest feature
                lateral = self.psp_modules(features[i])
            else:
                lateral = lateral_conv(features[i])
            laterals.append(lateral)
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False
            )
        
        # Apply FPN convolutions
        fpn_outs = []
        for i in range(len(laterals) - 1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        fpn_outs.append(laterals[-1])  # PSP output
        
        # Resize all to same size
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size, mode='bilinear', align_corners=False
            )
        
        # Concatenate and classify
        output = torch.cat(fpn_outs, dim=1)
        output = self.conv_seg(output)
        
        return output
