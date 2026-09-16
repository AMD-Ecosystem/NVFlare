# NVFlare on AMD Instinct (ROCm)

This branch enables NVIDIA FLARE (NVFlare) on AMD Instinct GPUs via ROCm/HIP.

## Overview

NVFlare runs on AMD Instinct GPUs (MI300X, gfx942; MI350X / MI355X, gfx950) on top of
PyTorch for ROCm. GPU compute rides the ROCm PyTorch build with no changes to model
code, training loops, or the NVFlare federation protocol.

The one enablement change is in `nvflare/fuel/utils/gpu_utils.py`: `GPUResourceManager`
now falls back to `rocm-smi` when `nvidia-smi` is absent, so it discovers AMD GPUs and
reports their memory correctly.

## Prerequisites

- AMD Instinct GPU: MI300X (gfx942) or MI350X / MI355X (gfx950)
- ROCm 10.0.0 or newer -- install per the public guide:
  https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html
- A PyTorch build for ROCm
- Python 3.12
- Ubuntu 24.04

## Install from source

```bash
git clone --single-branch --branch rocm-enablement \
  https://github.com/AMD-Ecosystem/NVFlare.git
cd NVFlare
pip install -e .
```

Once a formal AMD release publishes an `amd-nvflare` wheel to the public index, it can be
installed directly with `pip install amd-nvflare`.

## Verify

```python
from nvflare.fuel.utils import gpu_utils as g

print("has_rocm_smi:", g.has_rocm_smi())               # True on ROCm
print("GPU ids:     ", g.get_host_gpu_ids())            # e.g. [0]
print("VRAM MiB:    ", g.get_host_gpu_memory_total())   # e.g. [196592] on MI300X
```

## Resources

- Upstream NVFlare: https://github.com/NVIDIA/NVFlare
- AMD ROCm install guide: https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html
