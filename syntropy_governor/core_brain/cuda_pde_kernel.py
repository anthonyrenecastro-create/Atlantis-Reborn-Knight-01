import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger("SyntropyCUDAKernel")

_EXT_HANDLE = None
_EXT_LOAD_ATTEMPTED = False


def _can_build_cuda_extension() -> bool:
    if not torch.cuda.is_available():
        return False
    if os.getenv("CUDA_HOME"):
        return True
    # If nvcc is in PATH, extension can still build in many environments.
    return any(Path(p).joinpath("nvcc").exists() for p in os.getenv("PATH", "").split(os.pathsep))


def load_pde_extension() -> Optional[object]:
    global _EXT_HANDLE, _EXT_LOAD_ATTEMPTED

    if _EXT_HANDLE is not None:
        return _EXT_HANDLE
    if _EXT_LOAD_ATTEMPTED:
        return None

    _EXT_LOAD_ATTEMPTED = True

    if not _can_build_cuda_extension():
        logger.info("CUDA extension build prerequisites not found; using PyTorch fallback PDE step.")
        return None

    base = Path(__file__).resolve().parent
    cpp_path = base / "cuda_kernels" / "pde_kernel.cpp"
    cu_path = base / "cuda_kernels" / "pde_kernel.cu"

    if not cpp_path.exists() or not cu_path.exists():
        logger.warning("CUDA extension sources missing; using fallback PDE step.")
        return None

    try:
        _EXT_HANDLE = load(
            name="syntropy_pde_cuda",
            sources=[str(cpp_path), str(cu_path)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        logger.info("Loaded CUDA PDE extension successfully.")
        return _EXT_HANDLE
    except Exception as exc:
        logger.warning(f"Failed to build/load CUDA PDE extension: {exc}. Using fallback PDE step.")
        return None


def pde_step_cuda(
    phi1: torch.Tensor,
    phi5: torch.Tensor,
    Phi: torch.Tensor,
    alpha: float,
    zeta: float,
    gamma1: float,
    beta: float,
    omega: float,
    theta: float,
    gamma5: float,
    dt: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    ext = load_pde_extension()
    if ext is None:
        return None

    if not phi1.is_cuda or not phi5.is_cuda or not Phi.is_cuda:
        return None

    out = ext.pde_step(
        phi1,
        phi5,
        Phi,
        float(alpha),
        float(zeta),
        float(gamma1),
        float(beta),
        float(omega),
        float(theta),
        float(gamma5),
        float(dt),
    )
    return out[0], out[1]
