"""
QwT Calibration for MMDetection Models

This script performs end-to-end QwT calibration on MMDetection models:
1. Load baseline model and evaluate
2. Quantize model (W4/A4)
3. Warmup quantization observers
4. Generate compensation blocks via linear regression
5. Evaluate compensated model

Usage:
    python run_calibration.py \\
        --config path/to/mmdet_config.py \\
        --checkpoint path/to/checkpoint.pth \\
        --coco-root path/to/coco \\
        --calibration-samples 512 \\
        --device cuda \\
        --batch-size 8
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Generator

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from config import QwTConfig
from quantization import quant_model, set_quant_state
from qwt import generate_compensation_model
from utils import (
    evaluate_mmdet_model,
    load_warmup_state,
    print_evaluation_summary,
    run_warmup_pass,
    save_warmup_state,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='QwT Calibration for MMDetection'
    )
    
    # Model configuration
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to MMDetection config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--coco-root',
        type=str,
        required=True,
        help='Path to COCO dataset root directory'
    )
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=512,
        help='Number of samples for calibration (default: 512)'
    )
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=100,
        help='Number of samples for evaluation (default: 100)'
    )
    
    # Device configuration
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu). Note: MMDetection only supports CPU and CUDA.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    
    # Quantization configuration
    parser.add_argument(
        '--w-bit',
        type=int,
        default=4,
        choices=[2, 4, 8],
        help='Weight quantization bit-width (default: 4)'
    )
    parser.add_argument(
        '--a-bit',
        type=int,
        default=4,
        choices=[2, 4, 8],
        help='Activation quantization bit-width (default: 4)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    parser.add_argument(
        '--warmup-checkpoint',
        type=str,
        default=None,
        help='Path to warmup checkpoint (to skip warmup)'
    )
    parser.add_argument(
        '--no-save-warmup',
        action='store_true',
        help='Do not save warmup state'
    )
    
    # Evaluation options
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline evaluation'
    )
    parser.add_argument(
        '--skip-quantized-eval',
        action='store_true',
        help='Skip quantized (no compensation) evaluation'
    )
    
    args = parser.parse_args()

    # Validate paths immediately
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.coco_root).exists():
        logger.error(f"COCO root directory not found: {args.coco_root}")
        sys.exit(1)
        
    return args


def build_runner(args: argparse.Namespace) -> Runner:
    """Build MMDetection Runner from config."""
    cfg = Config.fromfile(args.config)
    
    # Update data root
    cfg.data_root = args.coco_root
    
    # Update test dataloader config to use the provided COCO root
    if 'test_dataloader' in cfg:
        cfg.test_dataloader.dataset.data_root = args.coco_root
        cfg.test_dataloader.dataset.ann_file = 'annotations/instances_val2017.json'
        cfg.test_dataloader.dataset.data_prefix = dict(img='val2017/')
        
        # Limit to eval_samples by using indices
        cfg.test_dataloader.dataset.indices = list(range(args.eval_samples))
        
        # Update batch size
        cfg.test_dataloader.batch_size = args.batch_size

    # Update evaluators to point to the correct annotation file
    if 'val_evaluator' in cfg:
        if isinstance(cfg.val_evaluator, dict):
            cfg.val_evaluator.ann_file = str(Path(args.coco_root) / 'annotations/instances_val2017.json')
        elif isinstance(cfg.val_evaluator, list):
            for eval_cfg in cfg.val_evaluator:
                eval_cfg.ann_file = str(Path(args.coco_root) / 'annotations/instances_val2017.json')
                
    if 'test_evaluator' in cfg:
        if isinstance(cfg.test_evaluator, dict):
            cfg.test_evaluator.ann_file = str(Path(args.coco_root) / 'annotations/instances_val2017.json')
        elif isinstance(cfg.test_evaluator, list):
            for eval_cfg in cfg.test_evaluator:
                eval_cfg.ann_file = str(Path(args.coco_root) / 'annotations/instances_val2017.json')

    # Set work_dir
    cfg.work_dir = args.output_dir
    
    # Set checkpoint
    cfg.load_from = args.checkpoint
    
    # Suppress verbose logs
    cfg.log_level = 'WARNING'
    
    # Build runner
    runner = Runner.from_cfg(cfg)
    
    # Explicitly load checkpoint to ensure it's loaded before we start
    if args.checkpoint:
        runner.load_checkpoint(args.checkpoint)
        
    return runner





def run_evaluation(
    step_name: str,
    model: torch.nn.Module,
    config: QwTConfig,
    skip: bool = False
) -> Optional[Dict[str, float]]:
    """Helper function to run evaluation and log results."""
    if skip:
        logger.info(f"{step_name}: Skipping evaluation\n")
        return None
        
    logger.info(f"{step_name}: Evaluating...")
    metrics = evaluate_mmdet_model(
        model, config.coco_root, config.eval_samples, config.device
    )
    logger.info(f"✓ {step_name} box mAP: {metrics['bbox_mAP']:.3f}\n")
    return metrics
    
def main():
    """Main calibration pipeline."""
    args = parse_args()
    
    # Create configuration
    config = QwTConfig(
        model_config=args.config,
        checkpoint=args.checkpoint,
        coco_root=args.coco_root,
        calibration_samples=args.calibration_samples,
        eval_samples=args.eval_samples,
        device=args.device,
        batch_size=args.batch_size,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        output_dir=args.output_dir,
        warmup_checkpoint=args.warmup_checkpoint,
        save_warmup=not args.no_save_warmup
    )
    
    logger.info("="*70)
    logger.info("QwT Calibration Pipeline - MMDetection")
    logger.info("="*70)
    # Step 1: Build Runner and Load Model
    logger.info("Step 1: Building MMDetection Runner...")
    
    # Suppress MMEngine logs
    logging.getLogger('mmengine').setLevel(logging.WARNING)
    
    runner = build_runner(args)
    model = runner.model
    model.cfg = runner.cfg
    model.runner = runner  # Attach runner for proper evaluation
    model.to(config.device)
    model.eval()
    logger.info("✓ Runner built and model loaded\n")
    
    # Step 2: Baseline evaluation
    baseline_metrics = run_evaluation(
        "Step 2: Baseline Evaluation (FP32)",
        model,
        config,
        skip=args.skip_baseline
    )
    
    # Step 3: Quantize model
    logger.info(f"Step 3: Quantizing model (W{config.w_bit}/A{config.a_bit})...")
    backbone = model.backbone
    
    # Prepare quantization parameters
    input_quant_params = {'n_bits': config.a_bit}
    weight_quant_params = {'n_bits': config.w_bit}
    
    q_backbone = quant_model(
        backbone,
        input_quant_params=input_quant_params,
        weight_quant_params=weight_quant_params
    )
    model.backbone = q_backbone
    logger.info("✓ Model quantized\n")
    
    # Step 4: Warmup quantization
    warmup_path = config.warmup_path
    if config.warmup_checkpoint and Path(config.warmup_checkpoint).exists():
        logger.info(f"Step 4: Loading warmup state from {config.warmup_checkpoint}...")
        load_warmup_state(q_backbone, config.warmup_checkpoint, config.device)
        
        # Run warmup pass to initialize compiled graph
        dummy_input = torch.randn(1, 3, 800, 800).to(config.device)
        run_warmup_pass(q_backbone, dummy_input)
        
        logger.info("✓ Warmup state loaded\n")
    else:
        if config.warmup_checkpoint:
            logger.warning(f"Warmup checkpoint {config.warmup_checkpoint} not found. Running warmup...")
        
        logger.info(f"Step 4: Warmup quantization ({config.calibration_samples} samples)...")
        
        # Warmup - use test_dataloader for calibration
        set_quant_state(q_backbone, input_quant=True, weight_quant=True)
        
        count = 0
        for data_batch in runner.test_dataloader:
            # Preprocess data
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, training=False)
            images = data['inputs']
            
            # Run through quantized backbone
            _ = q_backbone(images)
            
            count += images.shape[0]
            if count >= config.calibration_samples:
                break
        
        # Save warmup state
        if config.save_warmup:
            save_warmup_state(q_backbone, warmup_path)
        
        logger.info("✓ Warmup complete\n")
    
    # Step 5: Evaluate quantized model (no compensation)
    quantized_metrics = run_evaluation(
        "Step 5: Quantized Evaluation (No Compensation)",
        model,
        config,
        skip=args.skip_quantized_eval
    )
    
    # Step 6: Generate QwT compensation
    logger.info("Step 6: Generating QwT compensation...")
    
    # Create calibration data generator using test_dataloader
    def get_calib_data():
        count = 0
        for data_batch in runner.test_dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, training=False)
            images = data['inputs']
            yield images, None
            
            count += images.shape[0]
            if count >= config.calibration_samples:
                break
    
    calib_loader = get_calib_data()
    
    compensated_backbone, losses = generate_compensation_model(
        q_backbone,
        calib_loader,
        device=config.device,
        num_samples=config.calibration_samples
    )
    
    model.backbone = compensated_backbone
    logger.info("✓ Compensation model generated\n")
    
    # Step 7: Evaluate QwT compensated model
    # Ensure quantization is enabled for compensated model
    set_quant_state(compensated_backbone, input_quant=True, weight_quant=True)
    
    qwt_metrics = run_evaluation(
        "Step 7: QwT Compensated Evaluation",
        model,
        config,
        skip=False # Always evaluate final model
    )
    
    # Print summary
    if baseline_metrics and quantized_metrics and qwt_metrics:
        print_evaluation_summary(baseline_metrics, quantized_metrics, qwt_metrics)
    
    logger.info("="*70)
    logger.info("QwT Calibration Complete!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
