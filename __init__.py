"""
int4-omnixpu — 统一 INT4 加载器（wa4 + tint4 融合）。

- 加载 wa4 格式（对称 u4 + 组 scale）与 tint4 格式（非对称 u4 + per-block zp）
  的模型，统一走 omni_xpu_kernel 的 oneDNN INT4 GEMM 家族加速。
- 回退阶梯：w4a8（onednn_s8u4_gemm 缺失）→ w4a16 →
  wa4 模型 → 纯 python 反量化；tint4 模型 → torchao。
- kernel（omni_xpu_kernel）分 A/B 版且更新频繁，不自动安装，见 README。
- torchao 仅在 tint4 无 kernel 回退时需要，缺失时启动自动安装（tint4 同款）。

包含节点：int4XPUModelLoader（INT4 统一加载器）/ int4XPUModelQuantizer /
INT4XPU LoRA Loader / INT4XPU LoRA Stack。
"""
import logging

log = logging.getLogger("int4-omnixpu")

try:
    from .int4_xpu_install import ensure_installed
    _ok = ensure_installed()  # 缺依赖才真正安装；已装则直接返回
    if not _ok:
        log.info("torchao missing/old -> install attempted")
except Exception as _e:
    log.warning("torchao auto-install skipped: %r", _e)

from .int4_xpu_loader import NODE_CLASS_MAPPINGS as _L, NODE_DISPLAY_NAME_MAPPINGS as _LD
from .int4_xpu_quantizer import NODE_CLASS_MAPPINGS as _Q, NODE_DISPLAY_NAME_MAPPINGS as _QD
from .int4_xpu_lora_loader import NODE_CLASS_MAPPINGS as _LR, NODE_DISPLAY_NAME_MAPPINGS as _LRD
from .int4_xpu_lora_stack import NODE_CLASS_MAPPINGS as _LS, NODE_DISPLAY_NAME_MAPPINGS as _LSD

NODE_CLASS_MAPPINGS = {**_L, **_Q, **_LR, **_LS}
NODE_DISPLAY_NAME_MAPPINGS = {**_LD, **_QD, **_LRD, **_LSD}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
