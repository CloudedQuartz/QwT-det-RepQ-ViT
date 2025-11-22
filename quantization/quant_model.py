"""
Model quantization for Swin Transformer backbones.

This module replaces standard PyTorch layers with quantized versions:
- Conv2d → QuantConv2d (with 8-bit input quantization for patch embedding)
- Linear → QuantLinear (with channel-wise quantization for qkv, fc1, reduction)
- WindowAttention → QuantWindowAttention (with QuantMatMul for attention operations)

The quantization follows the RepQ-ViT approach for vision transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional, Tuple

from timm.models.swin_transformer import WindowAttention
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul


class QuantWindowAttention(WindowAttention):
    """
    Quantized WindowAttention that replaces matmuls with QuantMatMul.
    Inherits from timm's WindowAttention to maintain compatibility.
    """
    def __init__(self, *args, input_quant_params={}, weight_quant_params={}, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Force fused_attn to False to ensure we use the manual path with quantized matmuls
        self.fused_attn = False

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
        qkv_input_quant_params['channel_wise'] = True  # Match the behavior in quant_model()
        
        self.qkv = QuantLinear(self.qkv.in_features, self.qkv.out_features, qkv_input_quant_params, weight_quant_params)
        self.proj = QuantLinear(self.proj.in_features, self.proj.out_features, input_quant_params, weight_quant_params)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with quantized matmuls.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Manual attention path from timm (modified for quantization)
        q = q * self.scale
        
        # Quantized MatMul 1: q @ k.transpose
        # attn = q @ k.transpose(-2, -1)
        attn = self.matmul1(q, k.transpose(-2, -1))
        
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # Quantized MatMul 2: attn @ v
        # x = attn @ v
        x = self.matmul2(attn, v)

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def quant_model(model, input_quant_params={}, weight_quant_params={}):
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
            if 'patch_embed' in name: # timm uses patch_embed
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
            # timm naming: qkv, fc1, fc2, proj
            if 'qkv' in name or 'fc1' in name or 'reduction' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params) 
            else:   
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params) 
            new_m.weight.data = m.weight.data
            if m.bias is not None:
                new_m.bias.data = m.bias.data
            setattr(father_module, name[idx:], new_m)
            
        elif isinstance(m, WindowAttention):
            # Replace WindowAttention with QuantWindowAttention
            idx = idx + 1 if idx != 0 else idx
            
            # Create new QuantWindowAttention
            # We need to extract init args from the existing module
            # timm stores them in member vars
            new_m = QuantWindowAttention(
                dim=m.dim,
                num_heads=m.num_heads,
                # head_dim might not be stored directly, but we can infer or use default
                # In timm, head_dim is calculated. 
                # We can pass window_size, qkv_bias, etc.
                window_size=m.window_size,
                qkv_bias=m.qkv.bias is not None,
                attn_drop=m.attn_drop.p,
                proj_drop=m.proj_drop.p,
                # device/dtype?
                input_quant_params=input_quant_params,
                weight_quant_params=weight_quant_params
            )
            
            # Copy weights and buffers
            new_m.load_state_dict(m.state_dict(), strict=False)
            
            # Ensure fused_attn is False
            new_m.fused_attn = False
            
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
