"""
MMDetection QwT: Quantization without Tears for MMDetection Models

A clean, modular implementation of QwT post-training quantization with
learned compensation blocks for MMDetection models.

Example:
    >>> from mmdet_qwt import QwTConfig, run_calibration
    >>> config = QwTConfig(
    ...     model_config='path/to/config.py',
    ...     checkpoint='path/to/checkpoint.pth',
    ...     coco_root='path/to/coco',
    ...     device='cuda'
    ... )
    >>> # Run calibration via run_calibration.py script
"""

__version__ = '1.0.0'
__author__ = 'QwT-MMDet Contributors'

from .config import QwTConfig

__all__ = ['QwTConfig']
