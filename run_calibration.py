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
import logging
import torch
from pathlib import Path

from config import QwTConfig
from data import create_calibration_loader
from quantization import quant_model, set_quant_state
from qwt import generate_compensation_model
from utils import (
    load_mmdet_model,
    save_warmup_state,
    load_warmup_state,
    run_warmup_pass,
    evaluate_mmdet_model,
    print_evaluation_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
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
    
    return parser.parse_args()


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
    logger.info(f"\n{config}\n")
    
    # Step 1: Load model
    logger.info("Step 1: Loading MMDetection model...")
    model = load_mmdet_model(config.model_config, config.checkpoint, config.device)
    logger.info("✓ Model loaded successfully\n")
    
    # Step 2: Baseline evaluation
    baseline_metrics = None
    if not args.skip_baseline:
        logger.info("Step 2: Baseline Evaluation (FP32)...")
        baseline_metrics = evaluate_mmdet_model(
            model, config.coco_root, config.eval_samples, config.device
        )
        logger.info(f"✓ Baseline box mAP: {baseline_metrics['bbox_mAP']:.3f}\n")
    else:
        logger.info("Step 2: Skipping baseline evaluation\n")
    
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
    if Path(warmup_path).exists():
        logger.info(f"Step 4: Loading warmup state from {warmup_path}...")
        load_warmup_state(q_backbone, warmup_path, config.device)
        
        # Run warmup pass to initialize compiled graph
        dummy_input = torch.randn(1, 3, 800, 800).to(config.device)
        run_warmup_pass(q_backbone, dummy_input)
        
        logger.info("✓ Warmup state loaded\n")
    else:
        logger.info(f"Step 4: Warmup quantization ({config.calibration_samples} samples)...")
        
        # Create calibration loader
        calib_loader = create_calibration_loader(
            config.coco_root,
            config.calibration_samples,
            image_size=800,
            batch_size=config.batch_size,
            device=config.device
        )
        
        # Warmup
        set_quant_state(q_backbone, input_quant=True, weight_quant=True)
        for i, (images, _) in enumerate(calib_loader):
            _ = q_backbone(images)
            if (i + 1) * config.batch_size >= config.calibration_samples:
                break
        
        # Save warmup state
        if config.save_warmup:
            save_warmup_state(q_backbone, warmup_path)
        
        logger.info("✓ Warmup complete\n")
    
    # Step 5: Evaluate quantized model (no compensation)
    quantized_metrics = None
    if not args.skip_quantized_eval:
        logger.info("Step 5: Evaluating quantized model (no compensation)...")
        set_quant_state(q_backbone, input_quant=True, weight_quant=True)
        quantized_metrics = evaluate_mmdet_model(
            model, config.coco_root, config.eval_samples, config.device
        )
        logger.info(f"✓ Quantized box mAP: {quantized_metrics['bbox_mAP']:.3f}\n")
    else:
        logger.info("Step 5: Skipping quantized evaluation\n")
    
    # Step 6: Generate QwT compensation
    logger.info("Step 6: Generating QwT compensation...")
    
    # Reload calibration loader
    calib_loader = create_calibration_loader(
        config.coco_root,
        config.calibration_samples,
        image_size=800,
        batch_size=config.batch_size,
        device=config.device
    )
    
    compensated_backbone, losses = generate_compensation_model(
        q_backbone,
        calib_loader,
        device=config.device,
        num_samples=config.calibration_samples
    )
    
    model.backbone = compensated_backbone
    logger.info("✓ Compensation model generated\n")
    
    # Step 7: Evaluate QwT compensated model
    logger.info("Step 7: Evaluating QwT compensated model...")
    set_quant_state(compensated_backbone, input_quant=True, weight_quant=True)
    qwt_metrics = evaluate_mmdet_model(
        model, config.coco_root, config.eval_samples, config.device
    )
    logger.info(f"✓ QwT box mAP: {qwt_metrics['bbox_mAP']:.3f}\n")
    
    # Print summary
    if baseline_metrics and quantized_metrics:
        print_evaluation_summary(baseline_metrics, quantized_metrics, qwt_metrics)
    
    logger.info("="*70)
    logger.info("QwT Calibration Complete!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
