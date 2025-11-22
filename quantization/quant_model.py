"""
Model quantization for Swin Transformer backbones.

This module replaces standard PyTorch layers with quantized versions:
- Conv2d → QuantConv2d (with 8-bit input quantization for patch embedding)
- Linear → QuantLinear (with channel-wise quantization for qkv, fc1, reduction)
- WindowMSA → QuantWindowMSA (with QuantMatMul for attention operations)

The quantization follows the RepQ-ViT approach for vision transformers,
adapted for MMDetection's Swin Transformer implementation.
"""

from copy import deepcopy
from typing import Optional, Dict

import torch
import torch.nn as nn

# Import MMDetection's WindowMSA instead of timm's
try:
    from mmdet.models.backbones.swin import WindowMSA
except ImportError:
    # Fallback for older MMCV versions
    from mmcv.cnn.bricks.transformer import WindowMSA

from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul


class QuantWindowMSA(WindowMSA):
    """
    Quantized WindowMSA that replaces matmuls with QuantMatMul.
    Inherits from MMDetection's WindowMSA to maintain compatibility.
    
    Args:
        input_quant_params (dict, optional): Parameters for input quantization. Default: None
        weight_quant_params (dict, optional): Parameters for weight quantization. Default: None
    """
    def __init__(
        self, 
        *args, 
        input_quant_params: Optional[Dict] = None, 
        weight_quant_params: Optional[Dict] = None, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if input_quant_params is None:
            input_quant_params = {}
        if weight_quant_params is None:
            weight_quant_params = {}

        # Initialize QuantMatMul modules
        # matmul1: q @ k.T
        # matmul2: attn @ v
        
        # post-softmax quantization params (for matmul2)
        input_quant_params_matmul2 = deepcopy(input_quant_params)
        input_quant_params_matmul2['log_quant'] = True
        
        self.matmul1 = QuantMatMul(input_quant_params)
        self.matmul2 = QuantMatMul(input_quant_params_matmul2)
        
        # Replace qkv and proj with QuantLinear
        # IMPORTANT: qkv uses channel-wise input quantization (like fc1, reduction)
        # proj uses regular input quantization
        
        # qkv - channel-wise input quantization
        qkv_input_quant_params = deepcopy(input_quant_params)
        qkv_input_quant_params['channel_wise'] = True
        
        self.qkv = QuantLinear(
            self.qkv.in_features, 
            self.qkv.out_features, 
            qkv_input_quant_params, 
            weight_quant_params
        )
        self.proj = QuantLinear(
            self.proj.in_features, 
            self.proj.out_features, 
            input_quant_params, 
            weight_quant_params
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with quantized matmuls.
        
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None): mask with shape of (num_windows, Wh*Ww, Wh*Ww)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply scale
        q = q * self.scale
        
        # Quantized MatMul 1: q @ k.transpose
        attn = self.matmul1(q, k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # Quantized MatMul 2: attn @ v
        x = self.matmul2(attn, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def quant_model(
    model: nn.Module, 
    input_quant_params: Optional[Dict] = None, 
    weight_quant_params: Optional[Dict] = None
) -> nn.Module:
    """
    Recursively replace layers with quantized versions.
    
    Args:
        model: Model to quantize
        input_quant_params: Parameters for input quantization
        weight_quant_params: Parameters for weight quantization
    
    Returns:
        Quantized model
    """
    if input_quant_params is None:
        input_quant_params = {}
    if weight_quant_params is None:
        weight_quant_params = {}

    # input
    input_quant_params_embed = deepcopy(input_quant_params)
    input_quant_params_embed['n_bits'] = 8

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            # Root module or not found in dict yet (should not happen for submodules)
            continue
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            if 'patch_embed' in name:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params_embed,
                    weight_quant_params
                )
            else:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params,
                    weight_quant_params
                )
            new_m.weight.data = m.weight.data
            if m.bias is not None:
                new_m.bias.data = m.bias.data
            setattr(father_module, name[idx:], new_m)
            
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            # Channel-wise for qkv, fc1, reduction
            if 'qkv' in name or 'fc1' in name or 'reduction' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params) 
            else:   
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params) 
            new_m.weight.data = m.weight.data
            if m.bias is not None:
                new_m.bias.data = m.bias.data
            setattr(father_module, name[idx:], new_m)
            
        elif isinstance(m, WindowMSA):
            # Replace WindowMSA with QuantWindowMSA (MMDetection implementation)
            idx = idx + 1 if idx != 0 else idx
            
            # Create new QuantWindowMSA
            new_m = QuantWindowMSA(
                embed_dims=m.embed_dims,
                num_heads=m.num_heads,
                window_size=m.window_size,
                qkv_bias=m.qkv.bias is not None,
                qk_scale=None,  # Will use default
                attn_drop_rate=m.attn_drop.p,
                proj_drop_rate=m.proj_drop.p,
                input_quant_params=input_quant_params,
                weight_quant_params=weight_quant_params
            )
            
            # Copy weights and buffers
            new_m.load_state_dict(m.state_dict(), strict=False)
            
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model: nn.Module, input_quant: bool = False, weight_quant: bool = False) -> None:
    """Set quantization state for all quantized modules in the model."""
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
