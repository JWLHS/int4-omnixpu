"""
int4_xpu_lora_stack.py — INT4XPU LoRA Stack v1.3

v1.3: ADD — AIMDO 兼容防护（lora_policy 恒 normal 后无实际阻断，保留调用点）
v1.2: FIX — QKV 融合层 seen 条件化（与 loader v3.3 一致）：
  slice 有效 → (module, target) 三段注入；slice 无效(None) → module 去重
  （_make_bake_pre_hook / _resolve_qkv_slices 从 loader 导入，已含形状自适应修复）
v1.1: Multi-LoRA injection (≤8). Same bake rollback + dedup logic as Loader.
"""
import time, logging
import torch, torch.nn as nn
import folder_paths, comfy.utils
from .int4_xpu_loader import _is_quant_linear
from .int4_xpu_lora_common import (
    _wa4_reset_all_loras, _auto_detect_format, _convert_bfl_to_standard,
    _parse_raw_lora_sd, _get_accelerator_device, _rot_quarot_tensor,
    _resolve_with_alias,
)
from .int4_xpu_lora_loader import _resolve_qkv_slices, _make_bake_pre_hook

log = logging.getLogger("int4-LoRA-Stack")


class INT4XPULoRAStack:
    NAME = "INT4XPU LoRA Stack"
    CATEGORY = "int4"

    @classmethod
    def INPUT_TYPES(cls):
        # █ 原样保留 █
        inp = {"required": {"model": ("MODEL", {"tooltip": "From int4XPUModelLoader"})}, "optional": {}}
        for i in range(1, 9):
            inp["optional"][f"lora_name_{i}"] = (["None"] + folder_paths.get_filename_list("loras"),)
            inp["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        return inp

    @classmethod
    def IS_CHANGED(cls, model, **kwargs):
        # █ 原样保留（缓存触发逻辑，不动）█
        import random
        return tuple([random.random()] + [(kwargs.get(f"lora_name_{i}"), round(kwargs.get(f"strength_{i}", 1.0), 4)) for i in range(1, 9)])

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    def apply(self, model, **kwargs):
        # ★ v1.3 新增：AIMDO 活跃时跳过 LoRA（防 0xC0000005 崩溃）
        from .int4_xpu_aimdo import lora_policy
        if lora_policy() == "skip":
            _wa4_reset_all_loras(model)
            object.__setattr__(model.model, '_wa4_lora_needs_reset', False)
            return (model,)

        to_apply = []
        for i in range(1, 9):
            n = kwargs.get(f"lora_name_{i}")
            s = kwargs.get(f"strength_{i}", 1.0)
            if n is None or n == "None" or n == "": continue
            if abs(s) < 1e-5: continue
            p = folder_paths.get_full_path("loras", n)
            if p is None: log.warning(f"[int4 Stack] '{n}' not found"); continue
            to_apply.append((n, p, s))

        if getattr(model.model, '_wa4_lora_needs_reset', False):
            _wa4_reset_all_loras(model)
            object.__setattr__(model.model, '_wa4_lora_needs_reset', False)
        if not to_apply:
            log.info("[int4 Stack] ✓ no active LoRAs")
            return (model,)

        base_model = model.model
        while hasattr(base_model, '_orig_mod'): base_model = base_model._orig_mod
        quarot_enabled = bool(getattr(base_model, '_wa4_quarot_enabled', False))
        group_size = int(getattr(base_model, '_wa4_quarot_gs', 0))
        index = getattr(base_model, '_wa4_lora_index', None) or {}
        dev = _get_accelerator_device()
        cpu = torch.device("cpu")

        H = None
        if quarot_enabled and group_size > 0:
            from .int4_xpu_quarot import build_hadamard
            H = build_hadamard(group_size, device="cpu", dtype=torch.float32)

        if not getattr(model.model, '_wa4_detach_patched', False):
            # █ 原样保留（detach 清缓存逻辑，不动）█
            _orig_detach = model.detach
            def _wa4_detach(unpatch_all=True):
                _wa4_reset_all_loras(model)
                object.__setattr__(model.model, '_wa4_lora_needs_reset', True)
                return _orig_detach(unpatch_all)
            object.__setattr__(model, 'detach', _wa4_detach)
            object.__setattr__(model.model, '_wa4_detach_patched', True)

        total_t0 = time.perf_counter()
        total_aq, total_ab = 0, 0

        for lora_name, lora_path, strength in to_apply:
            t0 = time.perf_counter()
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            fmt = _auto_detect_format(lora_sd)
            if fmt == "bfl": lora_sd = _convert_bfl_to_standard(lora_sd)
            lora_data = _parse_raw_lora_sd(lora_sd)

            aq, ab, unmatched = 0, 0, 0
            for norm, info in lora_data.items():
                lora_type = info.get("type", "standard")
                is_qkv = norm.endswith(".attn.qkv") or norm.endswith(".attn1.qkv") or norm.endswith(".attn2.qkv")
                targets = _resolve_qkv_slices(index, norm) if is_qkv else [(norm, None)]
                layer_matched = False
                seen = set()
                for target_path, qkv_slice in targets:
                    modules = _resolve_with_alias(index, target_path)
                    if not modules: continue
                    for module in modules:
                        mid = id(module)
                        # ★ v1.2 条件化去重（与 loader v3.3 一致）
                        key = mid if qkv_slice is None else (mid, target_path)
                        if key in seen: continue
                        seen.add(key)
                        is_quant = _is_quant_linear(module)
                        is_linear = isinstance(module, nn.Linear)
                        if not is_quant and not is_linear: continue
                        self._pop_module_lora(module, lora_name)
                        if lora_type == "lokr":
                            w1 = info.get("lokr_w1"); w2 = info.get("lokr_w2")
                            if w1 is None or w2 is None: continue
                            self._inject_lokr(module, lora_name, w1, w2, info.get("alpha"), strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=not is_quant)
                        else:
                            down = info.get("down"); up = info.get("up")
                            if down is None or up is None: continue
                            self._inject_standard(module, lora_name, down, up, info.get("alpha"), strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=not is_quant)
                        if is_quant: aq += 1
                        else: ab += 1
                        layer_matched = True
                if not layer_matched: unmatched += 1

            elapsed = time.perf_counter() - t0
            parts = []
            if aq: parts.append(f"{aq}q")
            if ab: parts.append(f"{ab}b")
            log.info("[int4 Stack] %s | %s | s=%.2f | %.2fs%s",
                     lora_name, "+".join(parts) if parts else "0", strength, elapsed,
                     f" | {unmatched}u" if unmatched else "")
            total_aq += aq; total_ab += ab

            if not hasattr(model.model, '_wa4_loras'):
                object.__setattr__(model.model, '_wa4_loras', [])
            model.model._wa4_loras.append({"name": lora_name, "strength": strength, "path": lora_path})
            del lora_sd, lora_data

        log.info("[int4 Stack] ✓ %d LoRAs | %dq+%db | %.2fs",
                 len(to_apply), total_aq, total_ab, time.perf_counter() - total_t0)
        return (model,)

    @staticmethod
    def _pop_module_lora(module, lora_name):
        # █ 原样保留（baked delta 回滚，不动）█
        if _is_quant_linear(module):
            le = getattr(module, '_wa4_lora_entries', None)
            if le is not None:
                le.pop(lora_name, None)
                if len(le) == 0: object.__setattr__(module, '_wa4_lora_entries', None)
        bs = getattr(module, '_wa4_bake_state', None)
        if bs is None: return
        applied = bs.pop('_applied', None)
        if applied is not None and hasattr(module, 'weight') and module.weight is not None:
            for delta_cpu, sl, se in applied:
                try:
                    neg = (-delta_cpu).to(device=module.weight.device, dtype=module.weight.dtype)
                    if sl is not None and se is not None:
                        module.weight.data[sl:se].add_(neg)
                    else:
                        module.weight.data.add_(neg)
                except Exception: pass
        bs.pop(lora_name, None)
        bs.pop('_pending', None)
        hh = bs.pop('_hook_handle', None)
        if hh is not None:
            try: hh.remove()
            except Exception: pass

    def _inject_standard(self, module, lora_name, down, up, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False):
        # █ 原样保留 █
        A = down.to(cpu, torch.float16).clone()
        B = up.to(cpu, torch.float16).clone()
        if quarot_enabled and H is not None: A = _rot_quarot_tensor(A, H, group_size)
        rank = up.shape[1] if up.ndim >= 2 else 1
        mult = ((alpha_val / max(rank, 1)) if alpha_val else 1.0) * strength
        if bake:
            bs = getattr(module, '_wa4_bake_state', None)
            if bs is None: bs = {}; object.__setattr__(module, '_wa4_bake_state', bs)
            pending = bs.get('_pending')
            if pending is None: pending = []; bs['_pending'] = pending
            sl = qkv_slice[0] if qkv_slice else None
            se = qkv_slice[1] if qkv_slice else None
            pending.append((A, B, mult, sl, se))
            if '_hook_handle' not in bs:
                hook = module.register_forward_pre_hook(_make_bake_pre_hook(module))
                bs['_hook_handle'] = hook
        else:
            le = getattr(module, '_wa4_lora_entries', None)
            if le is None: le = {}; object.__setattr__(module, '_wa4_lora_entries', le)
            if qkv_slice is not None:
                sl, se = qkv_slice
                le.setdefault(lora_name, []).append(
                    (A, B[sl:se].contiguous().clone(), mult, sl, se))
            else:
                le.setdefault(lora_name, []).append((A, B, mult))

    def _inject_lokr(self, module, lora_name, w1, w2, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False):
        # █ 原样保留 █
        w1_c = w1.to(cpu, torch.float16).clone(); w2_c = w2.to(cpu, torch.float16).clone()
        delta = torch.kron(w1_c, w2_c)
        to2 = module.out_features if hasattr(module, "out_features") else module.weight.shape[0]
        ti2 = module.in_features if hasattr(module, "in_features") else module.weight.shape[1]
        if delta.shape[0] < to2: delta = delta.repeat((to2 + delta.shape[0] - 1) // delta.shape[0], 1)
        if delta.shape[0] > to2: delta = delta[:to2, :]
        if delta.shape[1] < ti2: delta = delta.repeat(1, (ti2 + delta.shape[1] - 1) // delta.shape[1])
        if delta.shape[1] > ti2: delta = delta[:, :ti2]
        if quarot_enabled and H is not None and delta.shape[1] % group_size == 0:
            delta = delta.to(dev); delta = _rot_quarot_tensor(delta, H, group_size); delta = delta.to(cpu).contiguous().clone()
        else: delta = delta.contiguous().clone()
        if bake:
            bs = getattr(module, '_wa4_bake_state', None)
            if bs is None: bs = {}; object.__setattr__(module, '_wa4_bake_state', bs)
            pending = bs.get('_pending')
            if pending is None: pending = []; bs['_pending'] = pending
            sl = qkv_slice[0] if qkv_slice else None
            se = qkv_slice[1] if qkv_slice else None
            pending.append(("delta", delta, strength, sl, se))
            if '_hook_handle' not in bs:
                hook = module.register_forward_pre_hook(_make_bake_pre_hook(module))
                bs['_hook_handle'] = hook
        else:
            le = getattr(module, '_wa4_lora_entries', None)
            if le is None: le = {}; object.__setattr__(module, '_wa4_lora_entries', le)
            if qkv_slice is not None:
                sl, se = qkv_slice
                le.setdefault(lora_name, []).append(
                    ("delta", delta[sl:se, :].contiguous().clone(), strength, sl, se))
            else:
                le.setdefault(lora_name, []).append(("delta", delta, strength))


NODE_CLASS_MAPPINGS = {"INT4XPULoRAStack": INT4XPULoRAStack}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4XPULoRAStack": "INT4XPU LoRA Stack (up to 8)"}

