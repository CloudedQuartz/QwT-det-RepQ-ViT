"""
Compensation Block for QwT.

This module implements the CompensationBlock that wraps quantized blocks
and applies learned linear compensation to reduce quantization error.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CompensationBlock(nn.Module):
    """
    Wrapper block that applies linear compensation to reduce quantization error.
    
    The compensation is learned via linear regression during calibration:
        compensated_out = quantized_block(x) + x @ W_comp + b_comp
    
    Where W_comp and b_comp are learned to minimize:
        ||full_precision_out - compensated_out||
    
    Args:
        block (nn.Module): The quantized block to wrap
        W (torch.Tensor): Weight matrix for compensation (C_in x C_out)
        b (torch.Tensor): Bias vector for compensation (C_out,)
        r2_score (float): R² score of the linear regression fit
        linear_init (bool, optional): Whether linear regression initialization was used. Default: True
        block_id (int, optional): Identifier for this block (for logging). Default: 0
        local_rank (int, optional): Local rank for distributed training. Default: 0
    """
    
    def __init__(
        self,
        block: nn.Module,
        W: torch.Tensor,
        b: torch.Tensor,
        r2_score: float,
        linear_init: bool = True,
        block_id: int = 0,
        local_rank: int = 0
    ):
        super().__init__()
        
        self.block = block
        self.block_id = block_id
        self.r2_score = r2_score
        
        # Register compensation parameters
        self.register_parameter('comp_weight', nn.Parameter(W.clone()))
        self.register_parameter('comp_bias', nn.Parameter(b.clone()))
        
        # Initialize compensation parameters
        if linear_init and (r2_score > 0):
            self.comp_weight.data.copy_(W)
            self.comp_bias.data.copy_(b)
            if local_rank == 0:
                logger.info(f'Block {block_id} using linear init (R²={r2_score:.3f})')
        else:
            nn.init.zeros_(self.comp_weight)
            nn.init.zeros_(self.comp_bias)
            if local_rank == 0:
                logger.info(f'Block {block_id} using zero init (R²={r2_score:.3f})')
    
    def forward(
        self,
        x: torch.Tensor,
        hw_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass with compensation.
        
        Args:
            x (torch.Tensor): Input tensor
            hw_shape (tuple, optional): (H, W) shape for attention blocks
        
        Returns:
            torch.Tensor: Compensated output tensor
        """
        # Pass through the wrapped block
        if hw_shape is not None:
            # MMDetection style (pass hw_shape to block)
            out = self.block(x, hw_shape)
        else:
            # Standard style
            out = self.block(x)
        
        # Apply linear compensation
        # out = out + x @ comp_weight + comp_bias
        compensation = x @ self.comp_weight + self.comp_bias
        out = out + compensation
        
        return out
    
    def __repr__(self) -> str:
        return (
            f"CompensationBlock(block_id={self.block_id}, "
            f"r2_score={self.r2_score:.3f}, "
            f"weight_shape={tuple(self.comp_weight.shape)}"
            f")"
        )
