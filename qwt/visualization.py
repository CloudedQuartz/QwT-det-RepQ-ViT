"""
Visualization module for QwT calibration.
"""

import logging
import os
from typing import List, Optional, Dict, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Helper class for generating visualizations during QwT calibration.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Visualizations will be saved to {self.output_dir}")
        
    def plot_error_distribution(
        self,
        error_before: torch.Tensor,
        error_after: torch.Tensor,
        block_id: int,
        layer_id: Optional[int] = None
    ):
        """
        Plot histogram of quantization error before and after compensation.
        
        Args:
            error_before: Tensor of errors (FP - Quant) before compensation
            error_after: Tensor of errors (FP - Compensated) after compensation
            block_id: Block identifier
            layer_id: Layer identifier (optional)
        """
        try:
            # Convert to numpy
            e_before = error_before.detach().cpu().numpy().flatten()
            e_after = error_after.detach().cpu().numpy().flatten()
            
            # Randomly sample if too large to speed up plotting
            max_samples = 100000
            if len(e_before) > max_samples:
                indices = np.random.choice(len(e_before), max_samples, replace=False)
                e_before = e_before[indices]
                e_after = e_after[indices]
            
            plt.figure(figsize=(10, 6))
            
            # Plot histograms
            plt.hist(e_before, bins=100, alpha=0.5, label='Before Compensation', color='red', density=True)
            plt.hist(e_after, bins=100, alpha=0.5, label='After Compensation', color='blue', density=True)
            
            # Calculate stats
            mse_before = np.mean(e_before**2)
            mse_after = np.mean(e_after**2)
            improvement = (1 - mse_after/mse_before) * 100 if mse_before > 0 else 0
            
            title = f"Block {block_id}"
            if layer_id is not None:
                title += f" (Layer {layer_id})"
            title += f"\nMSE Reduction: {improvement:.2f}%"
            
            plt.title(title)
            plt.xlabel("Quantization Error")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save
            filename = f"error_dist_block_{block_id:03d}.png"
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot error distribution for block {block_id}: {e}")
            plt.close()

    def plot_calibration_metrics(self, losses: Dict[str, List[float]]):
        """
        Plot calibration metrics (MSE loss) across blocks.
        
        Args:
            losses: Dictionary containing 'before' and 'after' loss lists
        """
        try:
            if not losses['before'] or not losses['after']:
                return
                
            blocks = range(len(losses['before']))
            
            plt.figure(figsize=(12, 6))
            
            plt.plot(blocks, losses['before'], 'r-', label='Before Compensation', marker='o', markersize=3, alpha=0.7)
            plt.plot(blocks, losses['after'], 'b-', label='After Compensation', marker='o', markersize=3, alpha=0.7)
            
            plt.title("Quantization Error (MSE) per Block")
            plt.xlabel("Block ID")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale often helps visualize differences better
            
            save_path = os.path.join(self.output_dir, "calibration_metrics.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot calibration metrics: {e}")
            plt.close()

    def plot_r2_scores(self, r2_scores: List[float]):
        """
        Plot R2 scores for each block.
        
        Args:
            r2_scores: List of R2 scores
        """
        try:
            if not r2_scores:
                return
                
            blocks = range(len(r2_scores))
            
            plt.figure(figsize=(12, 6))
            
            plt.plot(blocks, r2_scores, 'g-', label='R² Score', marker='o', markersize=3, alpha=0.7)
            
            plt.title("Linear Regression Fit Quality (R² Score) per Block")
            plt.xlabel("Block ID")
            plt.ylabel("R² Score")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.output_dir, "r2_scores.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot R2 scores: {e}")
            plt.close()

    def plot_weight_heatmap(self, W: torch.Tensor, block_id: int, layer_id: Optional[int] = None):
        """
        Plot heatmap of compensation weights.
        
        Args:
            W: Weight matrix (C_in x C_out)
            block_id: Block identifier
            layer_id: Layer identifier (optional)
        """
        try:
            W_np = W.detach().cpu().numpy()
            
            # If matrix is too large, downsample for visualization
            if W_np.shape[0] > 128 or W_np.shape[1] > 128:
                # Simple strided slicing
                step_row = max(1, W_np.shape[0] // 128)
                step_col = max(1, W_np.shape[1] // 128)
                W_np = W_np[::step_row, ::step_col]
            
            plt.figure(figsize=(8, 6))
            plt.imshow(W_np, cmap='viridis', aspect='auto')
            plt.colorbar(label='Weight Value')
            
            title = f"Compensation Weights Block {block_id}"
            if layer_id is not None:
                title += f" (Layer {layer_id})"
            plt.title(title)
            plt.xlabel("Output Channels (downsampled)" if W_np.shape[1] < W.shape[1] else "Output Channels")
            plt.ylabel("Input Channels (downsampled)" if W_np.shape[0] < W.shape[0] else "Input Channels")
            
            filename = f"weight_heatmap_block_{block_id:03d}.png"
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot weight heatmap for block {block_id}: {e}")
            plt.close()

    def plot_map_comparison(
        self,
        baseline_map: float,
        quantized_map: float,
        compensated_map: float
    ):
        """
        Plot mAP comparison bar chart.
        """
        try:
            labels = ['Baseline (FP32)', 'Quantized (No Comp)', 'QwT Compensated']
            values = [baseline_map, quantized_map, compensated_map]
            colors = ['gray', 'red', 'green']
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.title("mAP Comparison")
            plt.ylabel("mAP")
            plt.ylim(0, max(values) * 1.1)
            plt.grid(axis='y', alpha=0.3)
            
            save_path = os.path.join(self.output_dir, "map_comparison.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot mAP comparison: {e}")
            plt.close()

    def plot_detections(
        self,
        image: torch.Tensor,
        results: List[dict],
        phase: str,
        image_id: int,
        score_thr: float = 0.3
    ):
        """
        Plot image with bounding boxes.
        
        Args:
            image: Image tensor (C, H, W)
            results: List of detection results dicts
            phase: Phase name (e.g., 'baseline', 'quantized')
            image_id: Image identifier
            score_thr: Score threshold for displaying boxes
        """
        try:
            # Prepare image
            # Image is already denormalized (loaded from disk)
            # Just convert to numpy and transpose
            img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
            
            # Simple min-max normalization for visualization if values are outside [0, 1]
            if img_np.min() < 0 or img_np.max() > 1:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Ensure contiguous array
            img_np = np.ascontiguousarray(img_np)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img_np)
            
            ax = plt.gca()
            
            # Draw boxes
            for res in results:
                if res['score'] < score_thr:
                    continue
                
                bbox = res['bbox']
                x, y, w, h = bbox
                
                # Create rectangle
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = f"{res.get('category_id', 'obj')}: {res['score']:.2f}"
                plt.text(
                    x, y - 5, label,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1)
                )
            
            plt.axis('off')
            plt.title(f"Detections - {phase} (Image {image_id})")
            
            # Create phase directory
            phase_dir = os.path.join(self.output_dir, phase)
            os.makedirs(phase_dir, exist_ok=True)
            
            filename = f"det_{image_id}.png"
            save_path = os.path.join(phase_dir, filename)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot detections for image {image_id}: {e}")
            plt.close()
