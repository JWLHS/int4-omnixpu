"""Unload hook: make ComfyUI cleanup also release CPU models + int4 refs.

ComfyUI's ``unload_all_models()`` only walks GPU devices, so cleanup nodes and
buttons leave CPU-resident text encoders registered. The int4 model graph is
also pinned by this plugin's ``_norm_idx_cache``. This hook:

1. runs the original unload (GPU models);
2. frees cpu-device entries (text encoders/VAE on CPU);
3. clears the int4 lora-index cache and class-level prewarm refs;
4. gc + xpu/aimdo cache release + CRT heap release;
"""
import gc
import logging

log = logging.getLogger("int4-cleanup")

_orig_unload_all_models = None


def _heapmin():
    try:
        import ctypes
        for lib in ("ucrtbase", "msvcrt"):
            try:
                getattr(ctypes.windll, lib)._heapmin()
                break
            except Exception:
                continue
    except Exception:
        pass


def _aimdo_empty():
    try:
        from comfy_aimdo import control as ctrl
        ctrl.empty_xpu_allocator_cache(wait=True)
    except Exception:
        pass


def _clear_int4_refs():
    cleared = 0
    try:
        from .int4_xpu_lora_common import _norm_idx_cache
        cleared = len(_norm_idx_cache)
        _norm_idx_cache.clear()
    except Exception:
        pass
    try:
        from .int4_xpu_loader import INT4XPULinear
        INT4XPULinear._prewarm_target = None
        INT4XPULinear._prewarm_done = False
    except Exception:
        pass
    return cleared


def _release(deep=False):
    try:
        gc.collect()
        from comfy import model_management as mm
        mm.soft_empty_cache()
        _aimdo_empty()
        n = _clear_int4_refs()
        gc.collect()
        _heapmin()
        if deep:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
            )
        log.info("[int4-cleanup] released (int4 cache entries=%d)", n)
    except Exception:
        pass


def _unload_all_models():
    result = _orig_unload_all_models() if _orig_unload_all_models is not None else None
    try:
        import torch
        from comfy import model_management as mm
        mm.free_memory(1e30, torch.device("cpu"))
    except Exception:
        pass
    _release(deep=True)
    return result


def apply_cleanup_patch():
    global _orig_unload_all_models
    try:
        import comfy.model_management as mm
        if getattr(mm, "_int4xpu_unload_patched", False):
            return
        _orig_unload_all_models = mm.unload_all_models
        mm.unload_all_models = _unload_all_models
        mm._int4xpu_unload_patched = True
        log.info("[int4] cleanup hook active")
    except Exception as e:
        log.warning("[int4] cleanup hook apply failed: %r", e)
