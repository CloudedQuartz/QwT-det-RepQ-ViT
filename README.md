# QwT for MMDetection

A clean, modular implementation of **QwT (Quantization without Tears)** for MMDetection models. This package enables efficient post-training quantization with learned compensation blocks to recover accuracy loss from quantization.

## Features

- ✅ **Modular Design**: Clean separation of concerns (config, quantization, compensation, data loading)
- ✅ **Device Support**: CPU and CUDA (GPU) acceleration only
- ✅ **Configurable**: Easy parameter tuning via config class or CLI
- ✅ **Warmup State Persistence**: Save/load quantization observer state for fast re-runs
- ✅ **MMDetection Compatible**: Works with MMDetection 3.x models (Swin, ResNet, etc.)
- ✅ **Comprehensive Logging**: Detailed progress tracking and metrics

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.5+ with CUDA support (for GPU acceleration)
- MMDetection 3.3.0
- MMCV 2.1.0 (with C++ extensions compiled for CPU/CUDA)
- timm 0.9.0+ (for Swin Transformer models)

### Step 1: Create Virtual Environment (Python 3.10)

> [!IMPORTANT]
> **Python 3.10 Required**: This project requires Python 3.10. Verify your Python version before creating the virtual environment.

**Verify Python 3.10 is installed:**
```bash
# Check Python version
python3.10 --version
# Should output: Python 3.10.x
```

**Windows:**
```bash
# Create venv with Python 3.10
python3.10 -m venv .venv_qwt
# Or if python3.10 is your default Python:
python -m venv .venv_qwt

# Activate
.\.venv_qwt\Scripts\activate
```

**Linux/Mac:**
```bash
# Create venv with Python 3.10
python3.10 -m venv .venv_qwt

# Activate
source .venv_qwt/bin/activate
```

**Verify activation:**
```bash
# Should show Python 3.10.x
python --version
```

### Step 2: Install PyTorch

**For CPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 12.1 (Recommended for GPU Acceleration):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> [!IMPORTANT]
> **Intel XPU (Intel Arc/Flex/Max GPUs) Not Supported**: While PyTorch 2.5+ supports Intel XPU devices, **MMDetection and MMCV do not** because their C++ operators (RoIAlign, NMS, deformable convolutions, etc.) are only compiled for CPU and CUDA backends. Attempting to use Intel XPU will result in errors when these operators are called. If you only have Intel XPU hardware, use CPU mode instead.

### Step 3: Install MMCV (with C++ Extensions)

MMCV must be built from source to enable C++ operators required for MMDetection.

**Download MMCV Source:**
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0
```

**Windows (with Visual Studio 2019/2022):**
```bash
# Open "x64 Native Tools Command Prompt for VS 2019/2022"
set DISTUTILS_USE_SDK=1
set MMCV_WITH_OPS=1
pip install -e .
```

**Linux:**
```bash
export MMCV_WITH_OPS=1
pip install -e .
```

**Verify Installation:**
```python
from mmcv.ops import RoIAlign
print("MMCV C++ extensions loaded successfully!")
```

### Step 4: Install MMDetection and Dependencies

```bash
pip install mmdet==3.3.0
pip install -r requirements.txt
```

### Step 5: Clone MMDetection Repository

The config files for various models are needed from the MMDetection repository:

```bash
git clone https://github.com/open-mmlab/mmdetection.git mmdet_repo
```

This will create an `mmdet_repo` directory containing all model configs (e.g., `mmdet_repo/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py`).

### Step 6: Download COCO Dataset

1. Download COCO 2017 Val images and annotations from [cocodataset.org](https://cocodataset.org/#download)
2. Extract to a directory (e.g., `C:/data/coco2017/`)

Directory structure:
```
coco2017/
├── annotations/
│   └── instances_val2017.json
└── val2017/
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    └── ...
```

## Usage

### Basic Usage

```bash
python run_calibration.py \
    --config path/to/mmdet_config.py \
    --checkpoint path/to/checkpoint.pth \
    --coco-root path/to/coco2017 \
    --calibration-samples 512 \
    --eval-samples 100 \
    --device cpu \
    --batch-size 1
```

### Example: Swin-T Mask R-CNN

```bash
# Download config and checkpoint
mkdir -p checkpoints
cd checkpoints
# Download the 3x model (46.0 mAP baseline - MMDetection 3.x compatible)
# For Windows PowerShell:
Invoke-WebRequest -Uri "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth" -OutFile "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.pth"
# For Linux/Mac:
# wget https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth -O mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.pth
cd ..

# Run calibration
python run_calibration.py \
    --config mmdet_repo/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py \
    --checkpoint checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.pth \
    --coco-root C:/data/coco2017 \
    --calibration-samples 512 \
    --eval-samples 100 \
    --device cpu \
    --batch-size 1
```

### With CUDA (Recommended for Production)

```bash
python run_calibration.py \
    --config mmdet_repo/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py \
    --checkpoint checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.pth \
    --coco-root C:/data/coco2017 \
    --calibration-samples 512 \
    --device cuda \
    --batch-size 8
```

> [!TIP]
> **Performance Tip**: Use CUDA with batch size 8-16 for significantly faster calibration. CPU mode is functional but much slower, especially for 512+ calibration samples.

### Skipping Warmup (Using Saved State)

After the first run, warmup state is saved to `outputs/mask_rcnn_swin_t_3x_warmup.pth`. Subsequent runs can skip warmup:

```bash
python run_calibration.py \
    --config mmdet_repo/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py \
    --checkpoint checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.pth \
    --coco-root path/to/coco2017 \
    --warmup-checkpoint outputs/mask_rcnn_swin_t_3x_warmup.pth \
    --calibration-samples 512 \
    --device cuda
```

This significantly reduces runtime!

### Command-Line Options

```
Model Configuration:
  --config PATH          Path to MMDetection config file
  --checkpoint PATH      Path to model checkpoint

Dataset Configuration:
  --coco-root PATH       Path to COCO dataset root
  --calibration-samples N  Number of calibration samples (default: 512)
  --eval-samples N       Number of evaluation samples (default: 100)

Device Configuration:
  --device {cpu,cuda}    Device to use (default: cpu)
  --batch-size N         Batch size (default: 1)
                         Note: Larger batch sizes require more memory but are faster on GPU

Quantization Configuration:
  --w-bit {2,4,8}        Weight bit-width (default: 4)
  --a-bit {2,4,8}        Activation bit-width (default: 4)

Output Configuration:
  --output-dir PATH      Output directory (default: ./outputs)
  --warmup-checkpoint PATH  Path to warmup checkpoint (skip warmup)
  --no-save-warmup       Do not save warmup state

Evaluation Options:
  --skip-baseline        Skip baseline evaluation
  --skip-quantized-eval  Skip quantized evaluation
```

## Project Structure

```
mmdet_qwt/
├── config.py              # Configuration dataclass
├── qwt/
│   ├── core.py           # Main QwT compensation logic
│   └── compensation.py   # CompensationBlock class
├── quantization/
│   ├── quantizer.py      # Quantizer classes
│   ├── quant_modules.py  # Quantized layers
│   └── quant_model.py    # Model quantization
├── data/
│   └── loader.py         # COCO data loaders
├── utils/
│   └── model_utils.py    # Model loading/saving/evaluation
├── run_calibration.py    # Main CLI script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## How It Works

QwT (Quantization without Tears) reduces quantization error through learned linear compensation:

1. **Quantize Model**: Apply W4/A4 quantization to model
2. **Warmup Observers**: Initialize quantization observers with calibration images
3. **Learn Compensation**: For each block:
   - Measure quantization error: `Y = FP_output - Quantized_output`
   - Fit linear regression: `Y = W * input + b`
   - Wrap block with compensation: `output = quantized_block(x) + W*x + b`
4. **Evaluate**: Compensated model recovers significant accuracy

## Results

Example results on Swin-T Mask R-CNN 3x (100 samples):

```
Baseline (FP32):        box_mAP = 0.453
Quantized (W4/A4):      box_mAP = 0.417
QwT Compensated:        box_mAP = 0.422
QwT Improvement:        +0.006 mAP
```

With 512 calibration samples, improvements are typically larger (+0.005-0.010 mAP).

## Advanced Configuration

### Programmatic Usage

```python
from config import QwTConfig
from data import create_calibration_loader
from quantization import quant_model, set_quant_state
from qwt import generate_compensation_model
from utils import load_mmdet_model, evaluate_mmdet_model

# Create config
config = QwTConfig(
    model_config='path/to/config.py',
    checkpoint='path/to/checkpoint.pth',
    coco_root='path/to/coco',
    calibration_samples=512,
    device='cuda',
    batch_size=8
)

# Load and quantize model
model = load_mmdet_model(config.model_config, config.checkpoint, config.device)
q_backbone = quant_model(model.backbone, w_bit=4, a_bit=4)

# Create data loader
calib_loader = create_calibration_loader(
    config.coco_root,
    config.calibration_samples,
    batch_size=config.batch_size,
    device=config.device
)

# Generate compensation
compensated_backbone, losses = generate_compensation_model(
    q_backbone,
    calib_loader,
    device=config.device
)

# Evaluate
model.backbone = compensated_backbone
metrics = evaluate_mmdet_model(model, config.coco_root, 100, config.device)
print(f"QwT mAP: {metrics['bbox_mAP']:.3f}")
```

## Troubleshooting

### MMCV C++ Extensions Not Loading

**Issue**: `ImportError: cannot import name 'RoIAlign'`

**Solution**: Rebuild MMCV from source with `MMCV_WITH_OPS=1`

### Out of Memory (OOM)

**Issue**: GPU/XPU runs out of memory

**Solutions**:
- Reduce `--batch-size` (try 1, 2, 4)
- Reduce `--calibration-samples`
- Use CPU with `--device cpu`

### Slow Calibration

**Issue**: Calibration takes too long

**Solutions**:
- Use GPU (`--device cuda`) with higher batch size
- Save warmup state and reuse (`--warmup-checkpoint`)
- Reduce calibration samples for testing (use 512 for final run)

## Hardware Requirements

### Minimum Requirements (CPU Mode)

- **CPU**: Any modern x86_64 CPU with AVX2 support
- **RAM**: 16GB minimum (32GB recommended for larger models)
- **Storage**: 10GB for COCO dataset + model checkpoints
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+

### Recommended Requirements (CUDA Mode)

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
  - RTX 3060/3070: Can handle batch size 4-8
  - RTX 3080/3090/4090: Can handle batch size 16+
  - A100/H100: Optimal performance with batch size 32+
- **CUDA**: Version 11.8 or 12.1
- **RAM**: 16GB system RAM
- **Storage**: 10GB for dataset + checkpoints

### Unsupported Hardware

- **Intel XPU (Arc/Flex/Max GPUs)**: Not supported due to MMCV C++ operator limitations
- **AMD GPUs (ROCm)**: Not supported by MMCV
- **Apple Silicon (M1/M2/M3)**: Limited support (CPU mode only, MPS not supported by MMCV)

- **QwT Paper**: [Quantization without Tears](https://arxiv.org/abs/2106.08295)
- **MMDetection**: [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- **Original QwT-det**: [megvii-research/RepQ-ViT](https://github.com/megvii-research/RepQ-ViT)

## License

This implementation is based on the original QwT-det-RepQ-ViT code and follows the same license.

## Citation

If you use this code, please cite the original QwT paper:

```bibtex
@article{li2021qwt,
  title={Quantization without Tears: A Flexible Framework for Post-Training Quantization},
  author={Li, Yuhang and others},
  journal={arXiv preprint arXiv:2106.08295},
  year={2021}
}
```
