"""
Quantized neural network modules for QwT.

This module provides quantized versions of Conv2d, Linear, and MatMul operations
with configurable quantization parameters for both weights and activations.
"""

from copy import deepcopy
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import UniformQuantizer, LogSqrt2Quantizer


class QuantConv2d(nn.Conv2d):
    """
    Quantized Conv2d layer.
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        input_quant_params (dict, optional): Parameters for input quantization. Default: None
        weight_quant_params (dict, optional): Parameters for weight quantization. Default: None
    """
    def __init__(
        self,   
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
        bias: bool = True,
        input_quant_params: Optional[Dict] = None,
        weight_quant_params: Optional[Dict] = None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        if input_quant_params is None:
            input_quant_params = {}
        if weight_quant_params is None:
            weight_quant_params = {}

        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super().__repr__()
        return f"({s}input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"

    def set_quant_state(self, input_quant: bool = False, weight_quant: bool = False):
        """Set quantization state for input and weight."""
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization."""
        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

        return out


class QuantLinear(nn.Linear):
    """
    Quantized Linear layer.
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        input_quant_params (dict, optional): Parameters for input quantization. Default: None
        weight_quant_params (dict, optional): Parameters for weight quantization. Default: None
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_quant_params: Optional[Dict] = None,
        weight_quant_params: Optional[Dict] = None,
        bias: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)

        if input_quant_params is None:
            input_quant_params = {}
        if weight_quant_params is None:
            weight_quant_params = {}

        self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super().__repr__()
        return f"({s}input_quant={self.use_input_quant}, weight_quant={self.use_weight_quant})"

    def set_quant_state(self, input_quant: bool = False, weight_quant: bool = False):
        """Set quantization state for input and weight."""
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization."""
        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.linear(x, weight=w, bias=self.bias)

        return out


class QuantMatMul(nn.Module):
    """
    Quantized Matrix Multiplication module.
    
    Args:
        input_quant_params (dict, optional): Parameters for input quantization. Default: None
    """
    def __init__(self, input_quant_params: Optional[Dict] = None):
        super().__init__()

        if input_quant_params is None:
            input_quant_params = {}

        input_quant_params_matmul = deepcopy(input_quant_params)
        if 'log_quant' in input_quant_params_matmul:
            input_quant_params_matmul.pop('log_quant')
            self.quantizer_A = LogSqrt2Quantizer(**input_quant_params_matmul)
        else:
            self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)

        self.use_input_quant = False

    def __repr__(self):
        s = super().__repr__()
        return f"({s}input_quant={self.use_input_quant})"

    def set_quant_state(self, input_quant: bool = False, weight_quant: bool = False):
        """Set quantization state for input."""
        self.use_input_quant = input_quant

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization."""
        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)

        out = A @ B
        return out
