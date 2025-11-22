"""
Core QwT compensation model generation.

This module implements the main QwT (Quantization without Tears) algorithm
for generating compensation blocks that reduce quantization error through
learned linear transformations.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from .compensation import CompensationBlock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants
LINEAR_COMPENSATION_SAMPLES = 512


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature tensors (supports list of tensors/tuples)."""
    
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        return self.data_list[idx]


def enable_quant(module: nn.Module) -> None:
    """Enable quantization for all quantized layers in a module."""
    from quantization.quant_modules import QuantConv2d, QuantLinear, QuantMatMul
    for _, m in module.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(True, True)


def disable_quant(module: nn.Module) -> None:
    """Disable quantization for all quantized layers in a module."""
    from quantization.quant_modules import QuantConv2d, QuantLinear, QuantMatMul
    for _, m in module.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(False, False)


def gather_tensor_from_multi_processes(
    input: torch.Tensor,
    world_size: int
) -> torch.Tensor:
    """Gather tensors from multiple processes for distributed training."""
    if world_size == 1:
        return input
    
    # Synchronize
    if input.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Gather
    gathered_tensors = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, input)
    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    
    # Synchronize
    if input.device.type == 'cuda':
        torch.cuda.synchronize()
    
    return gathered_tensors


def solve_linear_regression(
    XtX: torch.Tensor,
    XtY: torch.Tensor,
    YtY: torch.Tensor,
    sum_Y: torch.Tensor,
    N: int,
    block_id: int = 0,
    world_size: int = 1,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Solve linear regression using accumulated statistics.
    
    Solves: W_comp*X + b = Y
    
    Args:
        XtX: X_aug^T @ X_aug where X_aug = [X, 1]
        XtY: X_aug^T @ Y
        YtY: Sum of all squared elements in Y
        sum_Y: Sum of Y across all samples (per channel)
        N: Total number of samples
        block_id: Block identifier for logging
        world_size: Number of processes (for distributed training)
        device: Device string
    
    Returns:
        W_comp: Compensation weight matrix (C_in x C_out)
        b: Compensation bias vector (C_out,)
        r2_score: R² score of the fit
    """
    # Gather from multiple processes if needed
    if world_size > 1:
        XtX = gather_tensor_from_multi_processes(XtX.unsqueeze(0), world_size).sum(dim=0)
        XtY = gather_tensor_from_multi_processes(XtY.unsqueeze(0), world_size).sum(dim=0)
        YtY = gather_tensor_from_multi_processes(YtY.unsqueeze(0), world_size).sum(dim=0)
        sum_Y = gather_tensor_from_multi_processes(sum_Y.unsqueeze(0), world_size).sum(dim=0)
        N = N * world_size
    
    # Add regularization for numerical stability
    lambda_reg = 1e-6 * torch.eye(XtX.size(0), device=device)
    lambda_reg[-1, -1] = 0  # Don't regularize bias term
    
    # Solve linear system
    try:
        W_overall = torch.linalg.solve(XtX + lambda_reg, XtY)
    except RuntimeError:
        logger.warning(f"Singular matrix in block {block_id}, using pseudo-inverse")
        W_overall = torch.linalg.pinv(XtX) @ XtY
    
    # Extract weight and bias
    W_comp = W_overall[:-1, :]
    b = W_overall[-1, :]
    
    # Calculate R² score
    # R² = 1 - SS_res / SS_tot
    SS_res = YtY - 2 * (W_overall.t() @ XtY).trace() + (W_overall.t() @ XtX @ W_overall).trace()
    SS_tot = YtY - (sum_Y * sum_Y).sum() / N
    r2_score = 1 - SS_res / (SS_tot + 1e-8)
    
    return W_comp, b, r2_score.item()


@torch.no_grad()
def generate_compensation_model(
    q_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    world_size: int = 1,
    num_samples: int = LINEAR_COMPENSATION_SAMPLES,
    batch_size: int = 1
) -> Tuple[nn.Module, Dict[str, list]]:
    """Generate QwT compensation model for quantized network.
    
    This function:
    1. Extracts features from the patch embedding layer
    2. For each block in each layer:
        a. Computes quantization error (FP - Quant)
        b. Fits linear regression to predict error from input
        c. Wraps block with CompensationBlock using learned parameters
    3. Handles downsampling between layers
    
    Args:
        q_model: Quantized model (e.g., Swin Transformer)
        train_loader: DataLoader for calibration images
        device: Device to use ('cpu' or 'cuda')
        world_size: Number of processes (for distributed training)
        num_samples: Number of samples for calibration
        batch_size: Batch size for internal dataloaders
    """
    logger.info("Starting QwT compensation model generation")
    
    losses = {'before': [], 'after': []}
    global_block_id = 0
    
    q_model.eval()
    q_model.to(device)
    
    # Step 1: Extract features from patch embedding
    output_t = []
    sample_count = 0
    
    logger.info("Collecting patch embedding outputs...")
    for i, (image, target) in enumerate(tqdm(train_loader, desc="Patch Embed")):
        image = image.to(device)
        
        # Run patch embed
        x = q_model.patch_embed(image)
        
        # MMDetection PatchEmbed returns (x, (H, W)) tuple
        H, W = None, None
        if isinstance(x, tuple):
            x, (H, W) = x
        
        # Get feature map size if not returned
        if H is None:
            if hasattr(q_model.patch_embed, 'grid_size'):
                H, W = q_model.patch_embed.grid_size
            else:
                # Infer from patch size
                if hasattr(q_model.patch_embed, 'patch_size'):
                    patch_size = q_model.patch_embed.patch_size
                elif hasattr(q_model.patch_embed, 'projection'):
                    patch_size = q_model.patch_embed.projection.kernel_size
                else:
                    patch_size = 4
                
                if isinstance(patch_size, int):
                    patch_size = (patch_size, patch_size)
                H = image.shape[2] // patch_size[0]
                W = image.shape[3] // patch_size[1]
        
        # Store as (tensor, (H, W))
        # x is (B, L, C)
        output_t.append((x.detach().cpu(), (H, W)))
        sample_count += image.size(0)
        if sample_count >= num_samples:
            break
    
    # Do NOT concatenate output_t, keep as list
    output_previous = output_t
    
    # Create dataset for features
    feature_set = FeatureDataset(output_previous)
    
    # Identify layers
    if hasattr(q_model, 'layers'):
        layers_list = q_model.layers
        logger.info(f"Using 'layers' attribute ({len(layers_list)} layers)")
    elif hasattr(q_model, 'stages'):
        layers_list = q_model.stages
        logger.info(f"Using 'stages' attribute ({len(layers_list)} stages)")
    else:
        raise AttributeError("Could not find 'layers' or 'stages' in model")

    # Step 2: Iterate through layers
    for layer_id, layer in enumerate(layers_list):
        logger.info(f"Processing Layer {layer_id}")
        
        # Apply downsample from previous layer BEFORE processing current layer blocks
        if layer_id > 0:
            prev_layer = layers_list[layer_id - 1]
            if hasattr(prev_layer, 'downsample') and prev_layer.downsample is not None:
                logger.info(f"  Applying downsample from Layer {layer_id-1} -> {layer_id}")
                
                output_downsampled_list = []
                ds_dataset = FeatureDataset(output_previous)
                ds_loader = torch.utils.data.DataLoader(
                    ds_dataset,
                    batch_size=batch_size, # batch_size=1 usually
                    collate_fn=lambda x: x[0] # Custom collate to return (tensor, (H, W)) directly
                )
                
                for t_out, (H, W) in ds_loader:
                    t_out = t_out.to(device)
                    H, W = int(H), int(W)
                    
                    # Reshape for downsample input
                    if len(t_out.shape) == 3:
                        B, L, C = t_out.shape
                        t_out_4d = t_out.view(B, H, W, C)
                        t_out_3d = t_out
                    else:
                        t_out_4d = t_out
                        B, H_temp, W_temp, C = t_out.shape
                        t_out_3d = t_out.view(B, -1, C)
                    
                    with torch.no_grad():
                        enable_quant(prev_layer.downsample)
                        # MMDetection PatchMerging: 3D input + input_size parameter
                        try:
                            out = prev_layer.downsample(t_out_3d, (H, W))
                        except TypeError:
                            out = prev_layer.downsample(t_out_4d)
                    
                    # Handle tuple return
                    if isinstance(out, tuple):
                        out = out[0]
                    
                    # Flatten to 3D if needed
                    if len(out.shape) == 4:
                        B, H_new, W_new, C_new = out.shape
                        out = out.view(B, -1, C_new)
                    
                    # Update H, W
                    H_new, W_new = (H + 1) // 2, (W + 1) // 2
                    output_downsampled_list.append((out.detach().cpu(), (H_new, W_new)))
                
                output_previous = output_downsampled_list
                
                # Update feature set
                feature_set.data_list = output_previous
        
        # Step 3: Process blocks in the layer
        for block_id, block in enumerate(layer.blocks):
            logger.info(f"  Processing Block {block_id} (Global {global_block_id})")
            
            # Determine dimensions
            dummy_input = output_previous[0][0].to(device)
            C_block = dummy_input.shape[-1]
            
            # Initialize accumulators for linear regression
            XtX = torch.zeros((C_block + 1, C_block + 1), device=device)
            XtY = torch.zeros((C_block + 1, C_block), device=device)
            YtY = torch.tensor(0.0, device=device)
            sum_Y = torch.zeros(C_block, device=device)
            
            # Create loader for current block
            feature_set.data_list = output_previous
            feature_loader = torch.utils.data.DataLoader(
                feature_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: x[0]
            )
            
            loss_before_accum = 0.0
            total_samples = 0
            
            # Pass 1: Accumulate statistics for linear regression
            for i, (t_out, (H, W)) in enumerate(tqdm(feature_loader, desc=f"Block {global_block_id} Stats", leave=False)):
                t_out = t_out.to(device)
                H, W = int(H), int(W)
                
                # Reshape to 3D and 4D
                if len(t_out.shape) == 4:
                    t_out_4d = t_out
                    B, H_cur, W_cur, C_cur = t_out.shape
                    t_out_3d = t_out.view(B, -1, C_cur)
                elif len(t_out.shape) == 3:
                    B, L, C_cur = t_out.shape
                    t_out_4d = t_out.view(B, H, W, C_cur)
                    t_out_3d = t_out
                
                # Run full precision
                disable_quant(block)
                with torch.no_grad():
                    try:
                        # MMDetection format (3D + hw_shape)
                        full_precision_out = block(t_out_3d, (H, W))
                    except (TypeError, ValueError):
                        try:
                            # timm format (4D)
                            full_precision_out = block(t_out_4d)
                        except:
                            # Fallback
                            full_precision_out = block(t_out_4d, (H, W))
                
                # Run quantized
                enable_quant(block)
                with torch.no_grad():
                    try:
                        quant_out = block(t_out_3d, (H, W))
                    except (TypeError, ValueError):
                        try:
                            quant_out = block(t_out_4d)
                        except:
                            quant_out = block(t_out_4d, (H, W))
                
                # Flatten to 3D
                if len(full_precision_out.shape) == 4:
                    full_precision_out = full_precision_out.view(B, -1, C_cur)
                if len(quant_out.shape) == 4:
                    quant_out = quant_out.view(B, -1, C_cur)
                
                # Compute quantization error
                X = t_out_3d.reshape(-1, C_cur)
                Y = (full_precision_out - quant_out).reshape(-1, C_cur)
                
                # Track loss before compensation
                loss_before_accum += Y.abs().sum().item()
                total_samples += Y.shape[0]
                
                # Augment X with bias term
                ones = torch.ones(X.shape[0], 1, device=device)
                X_aug = torch.cat([X, ones], dim=1)
                
                # Accumulate statistics
                XtX += X_aug.t() @ X_aug
                XtY += X_aug.t() @ Y
                YtY += (Y * Y).sum()
                sum_Y += Y.sum(dim=0)
            
            loss_before = loss_before_accum / total_samples
            
            # Solve linear regression
            W_comp, b, r2_score = solve_linear_regression(
                XtX, XtY, YtY, sum_Y, total_samples,
                block_id=global_block_id,
                world_size=world_size,
                device=device
            )
            
            logger.info(f'    Block {global_block_id}: abs_before={loss_before:.6f}, R²={r2_score:.3f}')
            losses['before'].append(loss_before)
            
            # Replace block with CompensationBlock
            comp_block = CompensationBlock(
                W=W_comp,
                b=b,
                r2_score=r2_score,
                block=block,
                linear_init=True,
                local_rank=0,
                block_id=global_block_id
            )
            layer.blocks[block_id] = comp_block.to(device)
            
            # Pass 2: Generate outputs for next block AND measure compensation effectiveness
            loss_after_accum = 0.0
            total_samples_after = 0
            output_previous_list = []
            
            for i, (t_out, (H, W)) in enumerate(feature_loader):
                t_out = t_out.to(device)
                H, W = int(H), int(W)
                
                # Reshape
                if len(t_out.shape) == 4:
                    t_out_4d = t_out
                    B, H_cur, W_cur, C_cur = t_out.shape
                    t_out_3d = t_out.view(B, -1, C_cur)
                elif len(t_out.shape) == 3:
                    B, L, C_cur = t_out.shape
                    t_out_4d = t_out.view(B, H, W, C_cur)
                    t_out_3d = t_out
                
                # Get full precision output for comparison
                disable_quant(block)
                with torch.no_grad():
                    try:
                        full_precision_out_after = block(t_out_3d, (H, W))
                    except (TypeError, ValueError):
                        try:
                            full_precision_out_after = block(t_out_4d)
                        except:
                            full_precision_out_after = block(t_out_4d, (H, W))
                
                # Get compensated output
                enable_quant(comp_block)
                with torch.no_grad():
                    try:
                        compensated_out = comp_block(t_out_3d, (H, W))
                    except (TypeError, ValueError):
                        try:
                            compensated_out = comp_block(t_out_4d)
                        except:
                            compensated_out = comp_block(t_out_4d, (H, W))
                
                # Flatten
                if len(full_precision_out_after.shape) == 4:
                    full_precision_out_after = full_precision_out_after.view(
                        full_precision_out_after.shape[0], -1, full_precision_out_after.shape[-1]
                    )
                if len(compensated_out.shape) == 4:
                    compensated_out_flat = compensated_out.view(
                        compensated_out.shape[0], -1, compensated_out.shape[-1]
                    )
                else:
                    compensated_out_flat = compensated_out
                
                # Calculate error after compensation
                Y_after = (full_precision_out_after - compensated_out_flat).reshape(-1, C_cur)
                loss_after_accum += Y_after.abs().sum().item()
                total_samples_after += Y_after.shape[0]
                
                # Flatten to 3D for next block
                if len(compensated_out.shape) == 4:
                    compensated_out = compensated_out.view(
                        compensated_out.shape[0], -1, compensated_out.shape[-1]
                    )
                
                output_previous_list.append((compensated_out.detach().cpu(), (H, W)))
            
            loss_after = loss_after_accum / total_samples_after
            losses['after'].append(loss_after)
            reduction = (1 - loss_after/loss_before) * 100 if loss_before > 0 else 0
            logger.info(f'    abs_after={loss_after:.6f}, reduction={reduction:.2f}%')
            
            output_previous = output_previous_list
            global_block_id += 1
            
            # Clean up
            del XtX, XtY, YtY, sum_Y, W_comp, b
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
    
    logger.info("Compensation model generation finished")
    return q_model, losses
