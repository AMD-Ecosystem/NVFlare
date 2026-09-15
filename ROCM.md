# nvflare on AMD Instinct -- Installation

This branch delivers the AMD ROCm port of NVIDIA FLARE (NVFlare) 2.8.1.

## Runs on AMD Instinct (ROCm 10.1+)

NVFlare runs on AMD Instinct GPUs (MI300X, MI350X/MI355X, MI450) via PyTorch-ROCm.
GPU compute rides the ROCm framework build with no changes to model code, training
loops, or the NVFlare federation protocol.

**The one change:** `nvflare/fuel/utils/gpu_utils.py` now tries `rocm-smi` when
`nvidia-smi` is absent, so `GPUResourceManager` correctly discovers AMD GPUs.

## Prerequisites

- AMD Instinct GPU: MI300X, MI350X / MI355X, MI450
- ROCm 10.1 or newer
- Python 3.12
- OS: manylinux2.28 (RHEL8.10) or Ubuntu 22.04

## Install (from AMD PyPI)

```bash
# 1. ROCm PyTorch (stable multi-arch index; arch selected by the extra)
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "rocm==10.1" "torch[device-gfx942]>=2.10.0" torchvision
#    (MI350X/MI355X: use torch[device-gfx950])

# 2. Install amd-nvflare
pip install amd-nvflare --index-url https://pypi.amd.com/rocm-10.1/simple

# 3. Remaining deps
pip install -r requirements.txt
```

## Install (from source / this branch)

```bash
# Reconstruct: clone upstream at the base + apply the 2 AMD files
# (see PORT_BASE.md for the full reconstruction recipe and repro scripts)
git clone --single-branch --branch main https://github.com/NVIDIA/NVFlare.git nvflare
cd nvflare
git checkout a700b64b3654
# Apply the AMD shim (from this branch):
#   nvflare/fuel/utils/gpu_utils.py
#   tests/unit_test/fuel/utils/gpu_utils_test.py
pip install -e .
```

## Verify

```python
import torch
print("HIP available:", torch.cuda.is_available())  # True on ROCm

from nvflare.fuel.utils import gpu_utils as g
print("has_rocm_smi:", g.has_rocm_smi())             # True
print("GPU ids:     ", g.get_host_gpu_ids())          # [0] on MI300X
print("VRAM MiB:    ", g.get_host_gpu_memory_total()) # [196592] on MI300X
```

Expected output (MI300X example):
```
HIP available: True
has_rocm_smi: True
GPU ids:      [0]
VRAM MiB:     [196592]
```

## Validated configurations

| GPU | Architecture | ROCm | Tests |
|-----|-------------|------|-------|
| AMD Instinct MI300X | gfx942 | 10.1 | 81/81 pass |
| AMD Instinct MI350X | gfx950 | 10.1 | 81/81 pass |
| AMD Instinct MI450 (FFM sim) | gfx1250 | 10.x | 2/2 pass |

## Resources

- Upstream NVFlare: https://github.com/NVIDIA/NVFlare
- AMD AIOSS release docs: `hipshift/release_docs/`
- Port artifacts: `hipshift/`
- Repro scripts: `hipshift/repro/`
