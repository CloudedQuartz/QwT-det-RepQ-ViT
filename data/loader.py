"""
Data loading utilities for QwT calibration.

This module provides data loaders for COCO dataset with fixed-size preprocessing
required for QwT calibration.
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QwTCalibrationLoader:
    """Data loader for QwT calibration with fixed-size images.
    
    This loader provides fixed-size 800x800 images for calibration,
    which is required for QwT's tensor concatenation operations.
    
    Args:
        coco_root: Path to COCO dataset root directory
        num_samples: Number of samples to load
        image_size: Target image size (default: 800)
        batch_size: Batch size (default: 1)
        device: Device to load images to
    """
    
    def __init__(
        self,
        coco_root: str,
        num_samples: int = 512,
        image_size: int = 800,
        batch_size: int = 1,
        device: str = 'cpu'
    ):
        self.coco_root = Path(coco_root)
        self.num_samples = num_samples
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        
        # Initialize COCO API
        ann_file = self.coco_root / 'annotations' / 'instances_val2017.json'
        self.coco = COCO(str(ann_file))
        
        # Get image IDs
        self.img_ids = list(self.coco.imgs.keys())[:num_samples]
        
        logger.info(f"Initialized QwTCalibrationLoader with {len(self.img_ids)} images")
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __iter__(self):
        """Iterate over calibration samples."""
        for i in range(0, len(self.img_ids), self.batch_size):
            batch_ids = self.img_ids[i:i+self.batch_size]
            images = []
            targets = []
            
            for img_id in batch_ids:
                # Load image
                img_info = self.coco.loadImgs(img_id)[0]
                img_path = self.coco_root / 'val2017' / img_info['file_name']
                
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to fixed size
                img = cv2.resize(img, (self.image_size, self.image_size))
                
                # Normalize (ImageNet stats)
                img = img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
                
                # Convert to CHW format
                img = np.transpose(img, (2, 0, 1))
                
                # Load annotations (dummy for calibration)
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                images.append(img)
                targets.append({'annotations': anns})
            
            if images:
                # Stack to batch tensor and convert to float32
                images_tensor = torch.from_numpy(np.stack(images)).float().to(self.device)
                yield images_tensor, targets


class COCOEvalDataset(Dataset):
    """COCO dataset for evaluation.
    
    This dataset provides images in their original aspect ratio for evaluation,
    compatible with MMDetection's aspect-ratio preserving resize.
    
    Args:
        coco_root: Path to COCO dataset root directory
        num_samples: Number of samples to use for evaluation
    """
    
    def __init__(
        self,
        coco_root: str,
        num_samples: int = 100
    ):
        self.coco_root = Path(coco_root)
        self.num_samples = num_samples
        
        # Initialize COCO API
        ann_file = self.coco_root / 'annotations' / 'instances_val2017.json'
        self.coco = COCO(str(ann_file))
        
        # Get image IDs
        self.img_ids = list(self.coco.imgs.keys())[:num_samples]
        
        logger.info(f"Initialized COCOEvalDataset with {len(self.img_ids)} images")
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, dict]:
        """Get image and annotations.
        
        Returns:
            image: RGB image array (H, W, 3)
            target: Dictionary with annotations
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.coco_root / 'val2017' / img_info['file_name']
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        target = {
            'image_id': img_id,
            'annotations': anns,
            'img_info': img_info
        }
        
        return img, target


def create_calibration_loader(
    coco_root: str,
    num_samples: int = 512,
    image_size: int = 800,
    batch_size: int = 1,
    device: str = 'cpu'
) -> QwTCalibrationLoader:
    """Create calibration data loader.
    
    Args:
        coco_root: Path to COCO dataset root directory
        num_samples: Number of calibration samples
        image_size: Target image size
        batch_size: Batch size
        device: Device to load data to
    
    Returns:
        Calibration data loader
    """
    return QwTCalibrationLoader(
        coco_root=coco_root,
        num_samples=num_samples,
        image_size=image_size,
        batch_size=batch_size,
        device=device
    )


def create_eval_dataset(
    coco_root: str,
    num_samples: int = 100
) -> COCOEvalDataset:
    """Create evaluation dataset.
    
    Args:
        coco_root: Path to COCO dataset root directory
        num_samples: Number of evaluation samples
    
    Returns:
        Evaluation dataset
    """
    return COCOEvalDataset(
        coco_root=coco_root,
        num_samples=num_samples
    )
