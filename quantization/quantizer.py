"""
Quantizer implementations for QwT.

This module provides two quantization strategies:
- UniformQuantizer: Asymmetric uniform quantization with learnable scale and zero-point
- LogSqrt2Quantizer: Logarithmic quantization for post-softmax attention values
"""

import torch
import torch.nn as nn
import numpy as np


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.channel_wise = channel_wise
        
        # Register as buffers so they're saved in state_dict
        self.register_buffer('inited', torch.tensor(False))
        self.register_buffer('delta', torch.zeros(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Dynamically resize buffers to match the saved state shape
        # This is necessary because channel-wise quantization changes the shape from [1] to [C, 1, 1, 1]
        for name in ['delta', 'zero_point']:
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if getattr(self, name).shape != input_param.shape:
                    getattr(self, name).resize_(input_param.shape)
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor):

        if not self.inited:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta = delta
            self.zero_point = zero_point
            self.inited = torch.tensor(True)

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    # Optimization: Downsample for large tensors on CPU
                    if x_clone.numel() > 10000 and x_clone.device.type == 'cpu':
                        # Random sampling for speed
                        indices = torch.randperm(x_clone.numel())[:10000]
                        sample = x_clone.reshape(-1)[indices]
                        new_max = torch.quantile(sample, pct)
                        new_min = torch.quantile(sample, 1.0 - pct)
                    else:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    # Fallback to numpy if torch.quantile fails or for other reasons
                    if x_clone.numel() > 10000:
                         indices = torch.randperm(x_clone.numel())[:10000]
                         sample = x_clone.reshape(-1)[indices].cpu().numpy()
                    else:
                         sample = x_clone.reshape(-1).cpu().numpy()
                         
                    new_max = torch.tensor(np.percentile(
                        sample, pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        sample, (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.channel_wise = channel_wise
        
        # Register as buffers so they're saved in state_dict
        self.register_buffer('inited', torch.tensor(False))
        self.register_buffer('delta', torch.zeros(1))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Dynamically resize buffers to match the saved state shape
        key = prefix + 'delta'
        if key in state_dict:
            input_param = state_dict[key]
            if self.delta.shape != input_param.shape:
                self.delta.resize_(input_param.shape)
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor):

        if not self.inited:
            delta = self.init_quantization_scale(x)
            self.delta = delta
            self.inited = torch.tensor(True)

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                if x_clone.numel() > 10000 and x_clone.device.type == 'cpu':
                    indices = torch.randperm(x_clone.numel())[:10000]
                    sample = x_clone.reshape(-1)[indices]
                    new_delta = torch.quantile(sample, pct)
                else:
                    new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                if x_clone.numel() > 10000:
                     indices = torch.randperm(x_clone.numel())[:10000]
                     sample = x_clone.reshape(-1)[indices].cpu().numpy()
                else:
                     sample = x_clone.reshape(-1).cpu().numpy()
                     
                new_delta = torch.tensor(np.percentile(
                    sample, pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0

        return x_float_q
