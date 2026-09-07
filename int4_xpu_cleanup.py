"""Cleanup dispatcher for int4 plugin.

ComfyUI calls ``unload_all_models()`` at the end of every prompt when
``--disable-smart-memory`` is active. That automatic call must keep the int4
XPU weights resident so repeated runs stay fast (old "Krea2 常驻保速度"
behaviour). Explicit cleanup entry points (buttons/nodes) must instead do a
full release: cpu-device text encoders, int4 global refs, aimdo pool and the
CRT heap.
"""
import gc
import logging
import traceback

log = logging.getLogger("int4-cleanup")

_orig_unload_all_models = None
_force_release = False
_auto_unload_in_progress = False
_plugin_used = False


def mark_plugin_used():
    global _plugin_used
    _plugin_used = True


def is_comfy_auto_unload():
    """True when unload_all_models comes from ComfyUI's prompt-end auto unload
    (execution.py DISABLE_SMART_MEMORY block, independent of wrapper order)."""
    try:
        for f in traceback.extract_stack(limit=12):
            if f.filename.replace("\\", "/").endswith("execution.py"):
                if 820 <= f.lineno <= 860:
                    return True
    except Exception:
        pass
    return False


def is_auto_unload_active():
    return _auto_unload_in_progress


def is_force_release_active():
    return _force_release


def _int4_present():
    """True only when this plugin actually loaded an int4 model in the session
    (prewarm target is kept while weights stay resident for warm reuse and
    cleared on explicit release)."""
    try:
        from .int4_xpu_loader import INT4XPULinear
        return _plugin_used or INT4XPULinear._prewarm_target is not None
    except Exception:
        return _plugin_used


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


def _release_resident_int4():
    """Force-release int4 XPU weights kept for warm reuse (auto-unload kept
    them resident and they are no longer in ComfyUI's current_loaded_models)."""
    try:
        from .int4_xpu_loader import INT4XPULinear, Int4LinearPython, Int4LinearTorchao
        target = INT4XPULinear._prewarm_target
        had_target = target is not None
        if target is not None:
            for m in target.modules():
                try:
                    if isinstance(m, INT4XPULinear):
                        m.release_xpu()
                    elif isinstance(m, (Int4LinearPython, Int4LinearTorchao)):
                        object.__setattr__(m, "_wa4_lora_gpu", None)
                except Exception:
                    pass
        INT4XPULinear._prewarm_target = None
        INT4XPULinear._prewarm_done = False
        return had_target
    except Exception:
        return False


def _release(deep=False):
    global _plugin_used
    active = _int4_present()
    released_resident = False
    try:
        gc.collect()
        from comfy import model_management as mm
        mm.soft_empty_cache()
        _aimdo_empty()
        if deep:
            released_resident = _release_resident_int4()
        n = _clear_int4_refs()
        gc.collect()
        _heapmin()
        if deep:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
            )
        if active or released_resident or n:
            log.info("[int4-cleanup] released (int4 cache entries=%d)", n)
            _plugin_used = False
    except Exception:
        pass


def _unload_all_models():
    global _force_release, _auto_unload_in_progress
    auto = is_comfy_auto_unload()
    if not _int4_present() or auto:
        # Prompt-end automatic unload: keep int4 weights resident for warm
        # reuse. For workflows that never used this plugin, just delegate to
        # ComfyUI unchanged.
        _auto_unload_in_progress = auto
        try:
            return _orig_unload_all_models() if _orig_unload_all_models is not None else None
        finally:
            _auto_unload_in_progress = False
    _force_release = True
    try:
        result = _orig_unload_all_models() if _orig_unload_all_models is not None else None
    finally:
        _force_release = False
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
        log.debug("[int4] cleanup hook active")
    except Exception as e:
        log.warning("[int4] cleanup hook apply failed: %r", e)
