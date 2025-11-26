"""
Model utilities for loading, saving, and evaluating models.

This module provides utilities for working with MMDetection models
in the QwT calibration pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from mmdet.apis import init_detector
from mmengine.runner import Runner
import mmcv

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
    
    # Filter to only load quantization buffers (delta, zero_point, inited)
    # This avoids overwriting model weights which might cause issues
    quant_state_dict = {
        k: v for k, v in state_dict.items() 
        if 'delta' in k or 'zero_point' in k or 'inited' in k
    }
    
    model.load_state_dict(quant_state_dict, strict=False)
    
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
    device: str = 'cpu',
    visualizer = None,
    phase: str = 'eval'
) -> Dict[str, float]:
    """Evaluate MMDetection model on COCO dataset using runner.
    
    Args:
        model: Model to evaluate (must have runner attribute)
        coco_root: Path to COCO dataset root
        num_samples: Number of samples to evaluate on
        device: Device to run evaluation on
        visualizer: Visualizer instance
        phase: Phase name for visualization
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model on {num_samples} samples...")
    
    if not hasattr(model, 'runner'):
        logger.error("Model does not have runner attribute")
        return {
            'bbox_mAP': 0.0, 'bbox_mAP_50': 0.0, 'bbox_mAP_75': 0.0,
            'bbox_mAP_small': 0.0, 'bbox_mAP_medium': 0.0, 'bbox_mAP_large': 0.0
        }
    
    runner = model.runner
    model.eval()
    
    # Collect results and image IDs
    results = []
    img_ids = []
    count = 0
    
    # Get the COCO API from the evaluator
    # We need to ensure the evaluator is initialized
    if not hasattr(runner.test_evaluator.metrics[0], '_coco_api'):
        runner.test_evaluator.metrics[0].dataset_meta = runner.test_dataloader.dataset.metainfo
        # Trigger lazy init if needed, or manually load
        # For now, assume it's initialized or we can access it via the dataset
        pass
        
    # Iterate through dataloader
    for data_batch in tqdm(runner.test_dataloader, desc="Evaluating", total=num_samples):
        with torch.no_grad():
            outputs = runner.model.test_step(data_batch)
            
        for output in outputs:
            img_id = output.img_id
            img_ids.append(img_id)
            
            # Convert predictions to COCO format
            pred_instances = output.pred_instances
            scores = pred_instances.scores
            bboxes = pred_instances.bboxes
            labels = pred_instances.labels
            
            for i in range(len(scores)):
                # COCO bbox format: [x, y, w, h]
                bbox = bboxes[i].tolist()
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y
                
                results.append({
                    'image_id': img_id,
                    'category_id': runner.test_dataloader.dataset.metainfo['classes'][labels[i].item()] if isinstance(labels[i].item(), str) else labels[i].item(), # Handle class mapping if needed, but usually label index matches if dataset is standard
                    # Actually, MMDetection labels are 0-indexed, COCO category IDs are not contiguous.
                    # We need to map label index to category ID.
                    # runner.test_dataloader.dataset.metainfo['classes'] gives class names.
                    # We need the category ID mapping.
                    # The dataset object usually has cat_ids.
                    'bbox': [x, y, w, h],
                    'score': scores[i].item()
                })

            # Visualize detections for this image
            if visualizer is not None and count < 5:
                try:
                    img_path = output.img_path
                    # Load original image
                    img = mmcv.imread(img_path)
                    # Convert BGR to RGB
                    img = mmcv.imconvert(img, 'bgr', 'rgb')
                    # Convert to tensor (C, H, W)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1)

                    # Filter results for this image
                    img_results = [r for r in results if r['image_id'] == img_id]

                    visualizer.plot_detections(
                        img_tensor,
                        img_results,
                        phase=phase,
                        image_id=img_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to visualize detections for {img_id}: {e}")
        
        count += len(outputs)
        if count >= num_samples:
            break
            
    # If no results, return 0
    if not results:
        return {
            'bbox_mAP': 0.0, 'bbox_mAP_50': 0.0, 'bbox_mAP_75': 0.0,
            'bbox_mAP_small': 0.0, 'bbox_mAP_medium': 0.0, 'bbox_mAP_large': 0.0
        }
        
    # Perform COCO evaluation
    try:
        # Get COCO GT object
        # Try to get it from the evaluator or dataset
        if hasattr(runner.test_evaluator.metrics[0], '_coco_api'):
            coco_gt = runner.test_evaluator.metrics[0]._coco_api
        else:
            # Fallback: load it manually (expensive but safe)
            from pycocotools.coco import COCO
            ann_file = runner.test_evaluator.metrics[0].ann_file
            coco_gt = COCO(ann_file)
            
        # Map label indices to category IDs
        # MMDetection dataset usually stores this mapping
        dataset = runner.test_dataloader.dataset
        if hasattr(dataset, 'cat_ids'):
             cat_ids = dataset.cat_ids
        else:
             # Assume 1-based mapping if not found (standard COCO)
             # But MMDetection usually handles this.
             # Let's try to get it from coco_gt
             cat_ids = coco_gt.getCatIds()
             
        # Update category IDs in results
        # MMDetection output labels are 0-indexed indices into the classes list
        # We need to map 0 -> cat_ids[0], 1 -> cat_ids[1], etc.
        for res in results:
            label_idx = res['category_id'] # This was the label index
            if label_idx < len(cat_ids):
                res['category_id'] = cat_ids[label_idx]
            else:
                # Fallback or error
                pass

        # Load results into COCO
        coco_dt = coco_gt.loadRes(results)
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = coco_eval.stats
        return {
            'bbox_mAP': metrics[0],
            'bbox_mAP_50': metrics[1],
            'bbox_mAP_75': metrics[2],
            'bbox_mAP_small': metrics[3],
            'bbox_mAP_medium': metrics[4],
            'bbox_mAP_large': metrics[5]
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'bbox_mAP': 0.0, 'bbox_mAP_50': 0.0, 'bbox_mAP_75': 0.0,
            'bbox_mAP_small': 0.0, 'bbox_mAP_medium': 0.0, 'bbox_mAP_large': 0.0
        }


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
