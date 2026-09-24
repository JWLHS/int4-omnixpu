"""显存不够时，把 int4 模型按 ComfyUI 正常路径卸下来（给下一个模型腾地方）。

为什么需要
----------
int4 层被本插件接管之后（模块替换 + 从 state_dict 里摘掉 weight），这些权重
**不在 ComfyUI 的记账里**：`ModelPatcherDynamic.loaded_size()` 只数得到模型里剩下
的非 int4 权重（Qwen Image 1.0 实测 42.6 MiB），OmniXPU 的 `DynamicVRAM boundary
trim` 用的就是这个数，于是会出现：

    VAE 解码要 5100 MiB，卡上只剩 3192 MiB，trim 只回收 202.6 MiB → 硬挤 → 撞驱动。

ComfyUI 自己也不会来卸：`free_memory()` 里 dynamic 模型之间互相不卸
（"don't actually unload dynamic models for the sake of other dynamic models as
that works on-demand"），而"按需"这个前提对插件自持的设备张量并不成立。

本模块做什么
------------
在 `comfy.model_management.load_models_gpu` 上挂一层。**只有同时满足**

  ① 场上还有**本次不加载**的 int4 模型占着设备，
  ② 这次要加载的模型不是"场上那个模型本身"（同模型重载无东西可让），且
  ③ 当前空闲显存 < 这次加载所需（min_inference / memory_required + reserved）

时，调用 ComfyUI 自己的 `free_memory(..., for_dynamic=False)`。
`for_dynamic=False` 正好绕过上面那条"dynamic 之间不互相卸"的规则，于是走的是
**既有、已实测**的卸载链：

    free_memory → LoadedModel.model_unload → ModelPatcher.detach
                → 插件 detach 包装 → INT4XPULinear.release_xpu()（逐层搬回内存）

模型被真正卸下（从 `current_loaded_models` 移除），下次用到时按正常加载路径重建。
解析/打包结果（`_w_packed` / `_w_s`）留在内存里，所以**热启动能力不变**，只多一次
H2D 上传（实测 Krea2 224 层 ≈ 1.2s、Qwen 839 层 ≈ 6s）。

为什么是"让 ComfyUI 卸"而不是"插件自己把张量搬走"
--------------------------------------------------
第一版实现是插件自己逐层 `release_xpu()`、**不告诉 ComfyUI**。同工作流实测
（cf 8191：采样 → 解码 → 再采样，同一模型同一 LoRA）：

  * 补丁关闭：`Prompt executed in 90.77s` 正常完成；
  * 补丁开启：第二次采样开始后进程卡死（CPU 不再增长、日志停住、GPU 仍占 12.3GB，
    也没有 GSC/驱动事件）。

原因是"模型仍留在 `current_loaded_models`、但设备张量已被搬走"这个中间状态
没经过任何已测路径，AIMDO 的 VBAR/换页记账仍以为它常驻。所以现在改成
**不动手搬张量，只让 ComfyUI 卸载** —— 卸载后的重载是插件一直在测的那条路。

安全边界（对应"不破坏现有功能"）
--------------------------------
* 任何异常都在内部吞掉，行为等价于没装本补丁（直接放行原函数）。
* 显存够用时一次都不动手（实测 Krea2 各 case `free=8316 MiB > target=3175 MiB`，
  补丁全程未触发）。
* 场上没有 int4 模型时不动手（原生工作流的卸载策略保持原样）。
* 正在加载的是我们自己的 int4 模型时绝不动手。
* 不额外 sync / 不额外 empty_cache（走 ComfyUI 既有卸载路径，含它自己的
  `soft_empty_cache`）。
* `OMNIXPU_INT4_YIELD_ON_PRESS=0` 一键回到原行为。
"""
from __future__ import annotations

import functools
import logging
import os

log = logging.getLogger("int4-omnixpu")

_PATCH_MARKER = "__int4xpu_yield_original__"
_OFF_VALUES = ("0", "false", "off", "no")


def _enabled() -> bool:
    return os.environ.get("OMNIXPU_INT4_YIELD_ON_PRESS", "1").strip().lower() not in _OFF_VALUES


def _int4_model_of(patcher):
    """patcher 由本插件加载过 int4 模型时返回它的 diffusion_model，否则 None。"""
    model = getattr(patcher, "model", None)
    dm = getattr(model, "diffusion_model", None)
    if dm is None:
        return None
    while hasattr(dm, "_orig_mod"):
        dm = dm._orig_mod
    options = getattr(patcher, "model_options", None) or {}
    try:
        ours = bool(options.get("wa4_int4")) or getattr(dm, "_wa4_lora_index", None) is not None
    except Exception:
        return None
    return dm if ours else None


def _expand(models):
    """与 load_models_gpu 内部一致：把挂在这些 patcher 上的模型也展开进来。"""
    out = []
    seen = set()
    pending = list(models or [])
    while pending:
        item = pending.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        out.append(item)
        try:
            pending.extend(item.model_patches_models())
        except Exception:
            pass
    return out


def _arg(args, kwargs, name, position, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _required_memory(model_management, args, kwargs) -> int:
    """这次加载大约需要多少空闲显存（与 OmniXPU 的 boundary trim 同一口径）。"""
    memory_required = _arg(args, kwargs, "memory_required", 0, 0) or 0
    minimum_required = None
    # comfy 0.36/0.37 的签名是 (models, memory_required, force_patch_weights,
    # minimum_memory_required, ...)；更老的版本 minimum_memory_required 在第 2 位。
    for position in (3, 2):
        candidate = _arg(args, kwargs, "minimum_memory_required", position, None)
        if candidate is not None and not isinstance(candidate, bool):
            minimum_required = candidate
            break
    required = max(int(memory_required), int(minimum_required or 0))
    inference = int(model_management.minimum_inference_memory())
    return max(inference, required + int(model_management.extra_reserved_memory()))


def _resident_int4(model_management, skip_ids, device) -> bool:
    """场上是否有还占着设备显存的 int4 模型（跳过本次要加载的那些）。"""
    for loaded in list(getattr(model_management, "current_loaded_models", []) or []):
        try:
            if loaded.is_dead():
                continue
            loaded_device = getattr(loaded, "device", None)
            if device is not None and loaded_device is not None and loaded_device != device:
                continue
            patcher = loaded.model
        except Exception:
            continue
        if id(patcher) in skip_ids:
            continue
        if _int4_model_of(patcher) is not None:
            return True
    return False


def _maybe_yield(model_management, models, args, kwargs) -> None:
    incoming = _expand(models)
    if not incoming:
        return

    device = None
    for item in incoming:
        candidate = getattr(item, "load_device", None)
        if candidate is not None and getattr(candidate, "type", None) == "xpu":
            device = candidate
            break
    if device is None:
        device = model_management.get_torch_device()
    if getattr(device, "type", None) != "xpu":
        return

    incoming_ids = {id(item) for item in incoming}
    if not _resident_int4(model_management, incoming_ids, device):
        # 场上没有"本次不加载"的 int4 模型：
        #   * 加载的就是场上的那个 int4 模型（重载/复用）→ 没东西可让，交给 ComfyUI；
        #   * 完全没有 int4 模型在场            → ComfyUI 自己的规则够用。
        return

    need = _required_memory(model_management, args, kwargs)
    free = int(model_management.get_free_memory(device))
    if free >= need:
        return

    keep_loaded = []
    for loaded in list(getattr(model_management, "current_loaded_models", []) or []):
        if id(getattr(loaded, "model", None)) in incoming_ids:
            keep_loaded.append(loaded)

    unloaded = model_management.free_memory(need, device, keep_loaded=keep_loaded, for_dynamic=False)
    if not unloaded or not log.isEnabledFor(logging.DEBUG):
        return

    names = []
    for loaded in unloaded:
        try:
            names.append(getattr(getattr(loaded, "model", None), "model", None).__class__.__name__)
        except Exception:
            pass
    incoming_name = ""
    for item in incoming:
        model = getattr(item, "model", None)
        if model is not None:
            incoming_name = model.__class__.__name__
            break
    log.debug("[int4] 卸载 %s → 加载 %s（需 %.0f MiB / 剩 %.0f MiB）",
              "/".join(names) or "模型", incoming_name or "下一个模型",
              need / (1024 ** 2), free / (1024 ** 2))


def apply_load_yield_patch() -> bool:
    """在 comfy.model_management.load_models_gpu 上挂一层（全局只挂一次）。"""
    try:
        import comfy.model_management as model_management
    except Exception as exc:
        log.debug("[int4] 卸载补丁跳过（import 失败）：%r", exc)
        return False

    original = getattr(model_management, "load_models_gpu", None)
    if original is None:
        return False
    if hasattr(original, _PATCH_MARKER):
        return False

    @functools.wraps(original)
    def load_models_gpu_with_yield(models, *args, **kwargs):
        if _enabled():
            try:
                _maybe_yield(model_management, models, args, kwargs)
            except Exception as exc:
                log.debug("[int4] 卸载检查失败（按原行为继续）：%r", exc)
        return original(models, *args, **kwargs)

    setattr(load_models_gpu_with_yield, _PATCH_MARKER, original)
    model_management.load_models_gpu = load_models_gpu_with_yield
    log.debug("[int4] 卸载补丁已启用")
    return True


__all__ = ["apply_load_yield_patch"]
