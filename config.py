"""
Configuration module for MMDetection QwT calibration.

This module provides a centralized configuration class for managing
all parameters related to QwT calibration, quantization, and evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class QwTConfig:
    """Configuration for QwT calibration on MMDetection models.
    
    Attributes:
        # Model Configuration
        model_config: Path to MMDetection config file
        checkpoint: Path to model checkpoint file
        
        # Dataset Configuration
        coco_root: Path to COCO dataset root directory
        warmup_samples: Number of samples for quantization warmup
        calibration_samples: Number of samples for QwT calibration
        eval_samples: Number of samples for evaluation
        
        # Device Configuration
        device: Device to use ('cpu', 'cuda')
                Note: XPU not supported as MMDetection requires CUDA-compiled MMCV operators
        batch_size: Batch size for calibration and evaluation
        
        # Quantization Configuration
        w_bit: Weight quantization bit-width
        a_bit: Activation quantization bit-width
        
        # Output Configuration
        output_dir: Directory for saving outputs (checkpoints, plots)
        warmup_checkpoint: Optional path to saved warmup state
        save_warmup: Whether to save warmup state after calibration
        
        # Visualization Configuration
        enable_plots: Whether to generate visualization plots
        plot_dir: Directory for saving plots
    """
    
    # Model Configuration
    model_config: str
    checkpoint: str
    
    # Dataset Configuration
    coco_root: str
    warmup_samples: int = 512
    calibration_samples: int = 512
    eval_samples: int = 100
    
    # Device Configuration
    device: Literal['cpu', 'cuda'] = 'cpu'
    batch_size: int = 1
    
    # Quantization Configuration
    w_bit: int = 4
    a_bit: int = 4
    
    # Output Configuration
    output_dir: str = './outputs'
    warmup_checkpoint: Optional[str] = None
    save_warmup: bool = True
    
    # Visualization Configuration
    enable_plots: bool = False
    plot_dir: str = './plots'
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert string paths to Path objects
        self.model_config = str(Path(self.model_config))
        self.checkpoint = str(Path(self.checkpoint))
        self.coco_root = str(Path(self.coco_root))
        self.output_dir = str(Path(self.output_dir))
        self.plot_dir = str(Path(self.plot_dir))
        
        # Validate device (MMDetection only supports CPU and CUDA)
        if self.device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {self.device}. MMDetection only supports 'cpu' or 'cuda'")
        
        # Validate bit-widths
        if self.w_bit not in [2, 4, 8]:
            raise ValueError(f"Invalid w_bit: {self.w_bit}. Must be 2, 4, or 8")
        if self.a_bit not in [2, 4, 8]:
            raise ValueError(f"Invalid a_bit: {self.a_bit}. Must be 2, 4, or 8")
        
        # Validate sample counts
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
        if self.eval_samples <= 0:
            raise ValueError("eval_samples must be positive")
        
        # Create output directories if they don't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.enable_plots:
            Path(self.plot_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def warmup_path(self) -> str:
        """Get the default warmup checkpoint path."""
        if self.warmup_checkpoint:
            return self.warmup_checkpoint
        # Default: outputs/model_name_warmup.pth
        model_name = Path(self.checkpoint).stem
        return str(Path(self.output_dir) / f"{model_name}_warmup.pth")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model_config': self.model_config,
            'checkpoint': self.checkpoint,
            'coco_root': self.coco_root,
            'calibration_samples': self.calibration_samples,
            'eval_samples': self.eval_samples,
            'device': self.device,
            'batch_size': self.batch_size,
            'w_bit': self.w_bit,
            'a_bit': self.a_bit,
            'output_dir': self.output_dir,
            'warmup_checkpoint': self.warmup_checkpoint,
            'save_warmup': self.save_warmup,
            'enable_plots': self.enable_plots,
            'plot_dir': self.plot_dir
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        lines = ["QwTConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}={value},")
        lines.append(")")
        return "\n".join(lines)
