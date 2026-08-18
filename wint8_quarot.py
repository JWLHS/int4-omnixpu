"""
wint8_quarot.py
───────────────
Group-wise Hadamard rotation for INT8 quantization quality improvement.

Spreads activation outliers across channels using orthogonal Hadamard matrices.
Based on QuaRot (2024) and ConvRot (2025) approaches, adapted for DiT models.

Originally from: https://github.com/newgrit1004/ComfyUI-ZImage-Triton (MIT)
Stripped-down for WINT8-Quantizer — pure PyTorch, scipy optional.
"""

import torch

try:
    from scipy.linalg import hadamard as _scipy_hadamard
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Cache Hadamard matrices by (size, device, dtype)
_HADAMARD_CACHE: dict = {}


def build_hadamard(
    size: int,
    device="cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a normalized orthogonal Hadamard matrix of the given size.
    Returns H such that H @ H^T = I.  Size must be a power of 2.

    Uses scipy if available (faster for large matrices), falls back to
    pure-PyTorch Sylvester construction.
    """
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size & (size - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")

    if _SCIPY_AVAILABLE:
        import numpy as np
        H_np = _scipy_hadamard(size)
        H = torch.tensor(H_np, dtype=dtype, device=device) / (size ** 0.5)
    else:
        # Pure-torch fallback: recursive Sylvester construction
        H = torch.tensor([[1.0]], dtype=dtype, device=device)
        n = 1
        while n < size:
            H = torch.cat([
                torch.cat([H,  H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
            n *= 2
        H = H / (size ** 0.5)

    _HADAMARD_CACHE[cache_key] = H
    return H


def rotate_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Rotate weight matrix offline:  W_rot = W @ H_block^T

    For Linear(in, out) with weight shape (out_features, in_features):
    each row of W is split into groups of group_size and rotated by H^T.

    Args:
        weight:     Shape (out_features, in_features).
        H:          Normalized Hadamard matrix, (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated weight, same shape as input.
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"in_features {in_f} not divisible by group_size {group_size}"
        )
    n_groups = in_f // group_size
    W_grouped = weight.view(out_f, n_groups, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


def rotate_activation(
    x: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Rotate activation online:  x_rot = x @ H_block

    Group-wise Hadamard spreads outliers across channels within each group.

    Args:
        x:          Shape (..., features).  Last dim divisible by group_size.
        H:          Normalized Hadamard matrix, (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated activation, same shape as input.
    """
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(
            f"features {features} not divisible by group_size {group_size}"
        )
    n_groups = features // group_size
    x_grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    H_dev = H.to(dtype=x.dtype, device=x.device)
    x_rot = torch.matmul(x_grouped, H_dev)
    return x_rot.view(orig_shape)
