"""
int4_xpu_lora_loader.py — INT4XPU LoRA Loader v3.5

v3.5: FIX — bake 纳入 AIMDO 管理：
  克隆新权重张量（AIMDO allocator 分配）→ 段加 delta → 整体换权重
  不再对已有权重页原地写（原地写绕过 AIMDO/VBAR 管理 → 0xC0000005 崩溃面）
v3.3: FIX — QKV 融合层通用修复：
  1. seen 去重条件化：slice 有效 → (module, target) 三段注入（融合层 q/k/v 都生效）
                      slice 无效(None) → module 去重（保留原版防重复注入）
  2. bake pre-hook 形状自适应：delta(3×head拼接) vs 独立head 自动分段
v3.2: FIX — bake pre-hook handles lokr "delta" entries.
v3.0: pre-hook bake (deferred GPU compute, no CPU stall).
"""
import time, logging
import torch, torch.nn as nn, re
import folder_paths, comfy.utils
from .int4_xpu_loader import _is_quant_linear
from .int4_xpu_lora_common import (
    _wa4_reset_all_loras, _auto_detect_format, _convert_bfl_to_standard,
    _parse_raw_lora_sd, _get_accelerator_device, _rot_quarot_tensor,
    _resolve_with_alias,
)

log = logging.getLogger("int4-LoRA")


def _resolve_qkv_slices(index, norm):
    """融合 QKV 三段切片：先按整层 out_features 判定，逐 target 兜底。"""
    base = norm.rsplit(".attn", 1)[0]
    for mod in _resolve_with_alias(index, norm):
        out_f = (mod.out_features if hasattr(mod, "out_features")
                 else (mod.weight.shape[0] if hasattr(mod, 'weight') and mod.weight is not None else 0))
        if out_f > 0 and out_f % 3 == 0:
            hs = out_f // 3
            return [(f"{base}.attn.wq", (0, hs)), (f"{base}.attn.wk", (hs, 2 * hs)), (f"{base}.attn.wv", (2 * hs, 3 * hs))]
    for probe in [f"{base}.attn.wq", f"{base}.attn.wk", f"{base}.attn.wv"]:
        matches = _resolve_with_alias(index, probe)
        if matches:
            mod = matches[0]
            out_f = (mod.out_features if hasattr(mod, "out_features")
                     else (mod.weight.shape[0] if hasattr(mod, 'weight') and mod.weight is not None else 0))
            if out_f > 0:
                return [(f"{base}.attn.wq", (0, out_f)), (f"{base}.attn.wk", (out_f, 2 * out_f)), (f"{base}.attn.wv", (2 * out_f, 3 * out_f))]
    return [(f"{base}.attn.wq", None), (f"{base}.attn.wk", None), (f"{base}.attn.wv", None)]


# ── Pre-hook for nn.Linear bake ──────────────────────────────

def _reset_bake_vbar(module):
    """Bake 替换权重后，清掉 ComfyUI 缓存的 vbar prefetch/签名，强制下次
    forward 用新权重重建 cast 路径。所有 bake 层都必须执行（权重都变了）。

    注意：_v_signature 只能置 None，不能 delattr ——
    comfy.ops.cast_modules_with_vbar 直接访问 s._v_signature，
    属性缺失会抛 AttributeError；置 None 后 vbar_signature_compare 返回
    False，强制走重建路径。
    """
    for attr in ("_prefetch", "_v_weight", "_v_bias"):
        try:
            if hasattr(module, attr):
                delattr(module, attr)
        except Exception:
            pass
    try:
        module._v_signature = None
    except Exception:
        pass


def _sync_baked_dtype(module, dtype):
    """仅对「实际权重 dtype ≠ 模型 act_dtype」的层同步归档 dtype（txtmlp 类
    fp32 未量化层），否则 ComfyUI geometry 按 fp32 算而 AIMDO vbar 缓冲按
    fp16 分配 → 2x 错位 → Buffer too small。正常 fp16 层不调用本函数，
    其他 lora 完全不受影响。
    """
    try:
        w = module.weight
        if w is not None:
            # 无条件设置：bake 换上的新权重是 clone 出来的，没有 _model_dtype，
            # hasattr 判断会漏掉 → geometry 退回 t.dtype(fp32)
            object.__setattr__(w, "_model_dtype", dtype)
            if hasattr(module, "weight_comfyn"):
                module.weight_comfyn = dtype
            if hasattr(module, "weight_comfy_model_dtype"):
                module.weight_comfy_model_dtype = dtype
            if hasattr(module, "bias_comfy_model_dtype"):
                module.bias_comfy_model_dtype = dtype
    except Exception:
        log.warning("[int4 LoRA] _sync_baked_dtype error", exc_info=True)


def _apply_bake_entry(w_new, entry, w_dtype):
    """Apply one bake entry onto the cloned weight; returns
    (delta_cpu_fp16, sl, se) for rollback, or None on shape mismatch."""
    if len(entry) >= 3 and isinstance(entry[0], str) and entry[0] == "delta":
        _, delta_cpu, mult = entry[:3]
        sl = entry[3] if len(entry) > 3 else None
        se = entry[4] if len(entry) > 4 else None
        delta_gpu = delta_cpu.to(device=w_new.device, dtype=w_dtype).mul_(mult)
    else:
        A_cpu, B_cpu, mult = entry[:3]
        sl = entry[3] if len(entry) > 3 else None
        se = entry[4] if len(entry) > 4 else None
        if sl is not None and se is not None and B_cpu.shape[0] != (se - sl):
            B_cpu = B_cpu[sl:se].contiguous()
        A_gpu = A_cpu.to(device=w_new.device, dtype=w_dtype)
        B_gpu = B_cpu.to(device=w_new.device, dtype=w_dtype)
        delta_gpu = (B_gpu @ A_gpu).mul_(mult)

    target_rows = (se - sl) if (sl is not None and se is not None) else w_new.shape[0]
    if delta_gpu.shape[0] == target_rows:
        if sl is not None and se is not None:
            w_new[sl:se] += delta_gpu
        else:
            w_new += delta_gpu
    elif delta_gpu.shape[0] % target_rows == 0:
        n = delta_gpu.shape[0] // target_rows
        base_sl = sl if sl is not None else 0
        for i in range(n):
            seg = slice(base_sl + i * target_rows, base_sl + (i + 1) * target_rows)
            w_new[seg] += delta_gpu[i * target_rows:(i + 1) * target_rows]
    else:
        log.warning(f"[int4 LoRA] shape mismatch delta={tuple(delta_gpu.shape)} "
                    f"target={tuple(w_new.shape)} — skip")
        return None
    return (delta_gpu.to(device=torch.device("cpu"), dtype=torch.float16).clone(), sl, se)


def _make_bake_pre_hook(module: nn.Module, act_dtype):
    def _pre_hook(_mod, _inputs):
        bs = getattr(module, '_wa4_bake_state', None)
        if bs is None: return
        pending = bs.get('_pending')
        if not pending: return
        # 仅当该层实际权重 dtype 与模型 act_dtype 不一致（典型：txtmlp 类
        # fp32 未量化层）才走 dtype 对齐；fp16 层与模型一致 → 完全跳过，
        # 不影响其他 lora / 其他层。
        # 兜底：act_dtype 缺失时按权重自身 dtype 处理（mismatch=False，
        # 走原 v3.5 行为，不做任何 dtype 转换）。
        target = act_dtype if act_dtype is not None else module.weight.dtype
        mismatch = module.weight.dtype != target
        w_dtype = module.weight.dtype
        applied = []
        try:
            # v3.5：克隆新权重（AIMDO 分配），全部 delta 在克隆张量上操作，最后整体换权重
            w_new = module.weight.detach().clone()
            for entry in pending:
                r = _apply_bake_entry(w_new, entry, w_dtype)
                if r is not None:
                    applied.append(r)
            if mismatch:
                # fp32 归档层 → bake 数学用原 dtype（fp32）算完，最后对齐
                # 模型 act_dtype，避免 AIMDO vbar(fp16) 与 ComfyUI
                # geometry(fp32) 2x 错位（Buffer too small）
                w_new = w_new.to(target)
            module.weight = nn.Parameter(w_new)   # 换新权重（AIMDO 分配），原页由 AIMDO 回收
            _reset_bake_vbar(module)
            if mismatch:
                _sync_baked_dtype(module, target)
        except Exception as e:
            log.warning("[int4 LoRA] bake pre-hook failed: %s", e)
        bs.pop('_pending', None)
        bs['_applied'] = applied
        hh = bs.pop('_hook_handle', None)
        if hh is not None: hh.remove()
    return _pre_hook


class INT4XPULoRALoader:
    NAME = "INT4XPU LoRA Loader"
    CATEGORY = "int4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "From int4XPUModelLoader"}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, model, lora_name, strength):
        # █ 原样保留（缓存触发逻辑，不动）█
        import random
        return (lora_name, strength, random.random())

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"

    def load_lora(self, model, lora_name, strength):
        if getattr(model.model, '_wa4_lora_needs_reset', False):
            _wa4_reset_all_loras(model)
            object.__setattr__(model.model, '_wa4_lora_needs_reset', False)
        if abs(strength) < 1e-5:
            self._remove_lora(model, lora_name)
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"[int4 LoRA] '{lora_name}' not found")

        # 同 LoRA 同强度已注入 → 直接跳过（避免每轮重建 entries + GPU 缓存，
        # 实测每轮重注入会让 XPU 分配器碎片化、显存逐轮 +~250MB）
        _prev = getattr(model.model, "_wa4_loras", None) or []
        if any(
            isinstance(x, dict)
            and x.get("name") == lora_name
            and x.get("path") == lora_path
            and abs(float(x.get("strength", 1.0)) - float(strength)) < 1e-5
            for x in _prev
        ):
            log.info("[int4 LoRA] ✓ %s 已按 strength=%.2f 注入，跳过重复注入",
                     lora_name, strength)
            return (model,)

        base_model = model.model
        while hasattr(base_model, '_orig_mod'): base_model = base_model._orig_mod
        quarot_enabled = bool(getattr(base_model, '_wa4_quarot_enabled', False))
        group_size = int(getattr(base_model, '_wa4_quarot_gs', 0))
        act_dtype = getattr(base_model, '_wa4_act_dtype', None)
        index = getattr(base_model, '_wa4_lora_index', None) or {}
        dev = _get_accelerator_device()
        cpu = torch.device("cpu")

        H = None
        if quarot_enabled and group_size > 0:
            from .int4_xpu_quarot import build_hadamard
            H = build_hadamard(group_size, device="cpu", dtype=torch.float32)

        t0 = time.perf_counter()
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        fmt = _auto_detect_format(lora_sd)
        if fmt == "bfl": lora_sd = _convert_bfl_to_standard(lora_sd)
        lora_data = _parse_raw_lora_sd(lora_sd)

        if not getattr(model.model, '_wa4_detach_patched', False):
            # █ 原样保留（detach 清缓存逻辑，不动）█
            _orig_detach = model.detach
            def _wa4_detach(unpatch_all=True):
                _wa4_reset_all_loras(model)
                object.__setattr__(model.model, '_wa4_lora_needs_reset', True)
                return _orig_detach(unpatch_all)
            object.__setattr__(model, 'detach', _wa4_detach)
            object.__setattr__(model.model, '_wa4_detach_patched', True)

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
                        self._inject_lokr(module, lora_name, w1, w2, info.get("alpha"), strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=not is_quant, act_dtype=act_dtype)
                    else:
                        down = info.get("down"); up = info.get("up")
                        if down is None or up is None: continue
                        self._inject_standard(module, lora_name, down, up, info.get("alpha"), strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=not is_quant, act_dtype=act_dtype)
                    if is_quant: aq += 1
                    else: ab += 1
                    layer_matched = True
            if not layer_matched: unmatched += 1

        elapsed = time.perf_counter() - t0
        parts = []
        if aq: parts.append(f"{aq} quant")
        if ab: parts.append(f"{ab} bake")
        if aq == 0 and ab == 0: parts = ["0 layers"]
        log.info("[int4 LoRA] ✓ %s | %s | strength=%s | %.2fs%s",
                 lora_name, " + ".join(parts), strength, elapsed,
                 f" | {unmatched} unmatched" if unmatched else "")

        if not hasattr(model.model, '_wa4_loras'):
            object.__setattr__(model.model, '_wa4_loras', [])
        model.model._wa4_loras.append({"name": lora_name, "strength": strength, "path": lora_path})
        del lora_sd, lora_data
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

    def _remove_lora(self, model, lora_name):
        # █ 原样保留 █
        bm = model.model
        while hasattr(bm, '_orig_mod'): bm = bm._orig_mod
        for m in bm.modules():
            self._pop_module_lora(m, lora_name)

    def _inject_standard(self, module, lora_name, down, up, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False, act_dtype=None):
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
                hook = module.register_forward_pre_hook(_make_bake_pre_hook(module, act_dtype))
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

    def _inject_lokr(self, module, lora_name, w1, w2, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False, act_dtype=None):
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
                hook = module.register_forward_pre_hook(_make_bake_pre_hook(module, act_dtype))
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


NODE_CLASS_MAPPINGS = {"INT4XPULoRALoader": INT4XPULoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4XPULoRALoader": "INT4XPU LoRA Loader"}

