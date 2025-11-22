"""
Model utilities for loading, saving, and evaluating models.

This module provides utilities for working with MMDetection models
in the QwT calibration pipeline.
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_mmdet_model(
    config_path: str,
    checkpoint_path: str,
    device: str = 'cpu'
) -> nn.Module:
    """Load MMDetection model.
    
    Args:
        config_path: Path to MMDetection config file
        checkpoint_path: Path to model checkpoint
        device: Device to load model to
    
    Returns:
        Loaded model
    """
    logger.info(f"Loading MMDetection model from {checkpoint_path}")
    model = init_detector(config_path, checkpoint_path, device=device)
    logger.info("Model loaded successfully")
    return model


def save_warmup_state(
    model: nn.Module,
    save_path: str
) -> None:
    """Save warmup state (quantization observer parameters).
    
    Args:
        model: Model with initialized quantization observers
        save_path: Path to save the state
    """
    logger.info(f"Saving warmup state to {save_path}")
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save state dict (includes quantization buffers)
    torch.save(model.state_dict(), save_path)
    
    logger.info("Warmup state saved successfully")


def load_warmup_state(
    model: nn.Module,
    load_path: str,
    device: str = 'cpu'
) -> nn.Module:
    """Load warmup state (quantization observer parameters).
    
    Args:
        model: Model to load state into
        load_path: Path to load the state from
        device: Device to load to
    
    Returns:
        Model with loaded warmup state
    """
    logger.info(f"Loading warmup state from {load_path}")
    
    # Load state dict
    state_dict = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    
    logger.info("Warmup state loaded successfully")
    return model


def run_warmup_pass(
    model: nn.Module,
    dummy_input: torch.Tensor
) -> None:
    """Run a dummy forward pass to initialize compiled graph.
   
    Args:
        model: Model to run warmup on
        dummy_input: Dummy input tensor
    """
    logger.info("Running warmup pass to initialize compiled graph...")
    model.eval()
    with torch.no_grad():
        _ = model.patch_embed(dummy_input)
    logger.info("Warmup pass complete")


def evaluate_mmdet_model(
    model: nn.Module,
    coco_root: str,
    num_samples: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate MMDetection model on COCO dataset.
    
    Args:
        model: Model to evaluate
        coco_root: Path to COCO dataset root
        num_samples: Number of samples to evaluate on
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model on {num_samples} samples...")
    
    coco_root = Path(coco_root)
    ann_file = coco_root / 'annotations' / 'instances_val2017.json'
    coco_gt = COCO(str(ann_file))
    
    img_ids = list(coco_gt.imgs.keys())[:num_samples]
    
    results = []
    model.eval()
    
    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = coco_root / 'val2017' / img_info['file_name']
        
        # Run inference
        result = inference_detector(model, str(img_path))
        
        # Convert to COCO format
        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances.cpu()
            bboxes = pred_instances.bboxes.numpy()
            scores = pred_instances.scores.numpy()
            labels = pred_instances.labels.numpy()
            
            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                
                results.append({
                    'image_id': img_id,
                    'category_id': int(label) + 1,  # COCO 1-indexed
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })
    
    # Run COCO evaluation
    if results:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'bbox_mAP': coco_eval.stats[0],
            'bbox_mAP_50': coco_eval.stats[1],
            'bbox_mAP_75': coco_eval.stats[2],
            'bbox_mAP_small': coco_eval.stats[3],
            'bbox_mAP_medium': coco_eval.stats[4],
            'bbox_mAP_large': coco_eval.stats[5]
        }
    else:
        logger.warning("No detection results, returning zero metrics")
        metrics = {
            'bbox_mAP': 0.0,
            'bbox_mAP_50': 0.0,
            'bbox_mAP_75': 0.0,
            'bbox_mAP_small': 0.0,
            'bbox_mAP_medium': 0.0,
            'bbox_mAP_large': 0.0
        }
    
    return metrics


def print_evaluation_summary(
    baseline_metrics: Dict[str, float],
    quantized_metrics: Dict[str, float],
    qwt_metrics: Dict[str, float]
) -> None:
    """Print evaluation summary.
    
    Args:
        baseline_metrics: Baseline (FP32) metrics
        quantized_metrics: Quantized (no compensation) metrics
        qwt_metrics: QwT compensated metrics
    """
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"  Baseline (FP32):        box_mAP = {baseline_metrics['bbox_mAP']:.3f}")
    print(f"  Quantized (W4/A4):      box_mAP = {quantized_metrics['bbox_mAP']:.3f}")
    print(f"  QwT Compensated:        box_mAP = {qwt_metrics['bbox_mAP']:.3f}")
    print(f"  QwT Improvement:        +{qwt_metrics['bbox_mAP'] - quantized_metrics['bbox_mAP']:.3f} mAP")
    print("="*70 + "\n")
