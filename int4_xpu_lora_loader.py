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
from .int4_xpu_loader import (
    _is_quant_linear, _wa4_lora_replay_tick, _wa4_lora_replay_drop,
)
from .int4_xpu_lora_common import (
    _wa4_reset_all_loras, _auto_detect_format, _convert_bfl_to_standard,
    _parse_raw_lora_sd, _get_accelerator_device, _rot_quarot_tensor,
    _resolve_with_alias, _wa4_lora_state_live,
)

log = logging.getLogger("int4-LoRA")

# 自动补回（卸载重放）进行中时的统计容器：重放期间每 LoRA 的注入明细降到
# DEBUG，只在结束时打一条汇总；正常节点注入时该值为 None。
_WA4_REPLAY_CTX = None


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

def _make_bake_pre_hook(module: nn.Module):
    def _pre_hook(_mod, _inputs):
        # 卸载重放要先跑：本层的 bake 条目可能正是重放刚装进来的
        _wa4_lora_replay_tick(module)
        bs = getattr(module, '_wa4_bake_state', None)
        if bs is None: return
        pending = bs.get('_pending')
        if not pending: return
        w_dev = module.weight.device
        w_dtype = module.weight.dtype
        cpu = torch.device("cpu")
        # pending: {lora_name: [entry, ...]} —— 同一层挂多个 LoRA 时按名字分开
        # 记账，这样移除其中一个只回滚它自己的 delta（旧结构是平铺 list，做
        # 防御性兼容处理）。
        if isinstance(pending, list):
            pending = {"": pending}
        pairs = [(ln, e) for ln, lst in pending.items() for e in lst]
        applied = {}
        try:
            from .int4_xpu_aimdo import aimdo_active as _aimdo_active
            _aimdo_on = _aimdo_active()
        except Exception:
            _aimdo_on = False
        try:
            if _aimdo_on:
                # ── AIMDO 路径：CPU 合成 delta + dtype 对齐 ──
                # AIMDO 生效时 GPU 小 GEMM（B@A）慢 100-1000 倍（实测 2-6s/层），
                # 且懒加载未量化层可能按文件原 dtype（F32）物化导致 vbar 缓冲错位
                # （Buffer too small）。这里在 CPU 合成 delta（32 层 <1s），换权时
                # 强制对齐到 weight_comfy_model_dtype（act_dtype），一并规避两者。
                target_dtype = getattr(module, "weight_comfy_model_dtype", None) or w_dtype
                w_new = module.weight.detach().clone().to(target_dtype)
                for _lname, entry in pairs:
                    if len(entry) >= 3 and isinstance(entry[0], str) and entry[0] == "delta":
                        _, delta_cpu, mult = entry[:3]
                        sl = entry[3] if len(entry) > 3 else None
                        se = entry[4] if len(entry) > 4 else None
                        delta_c = delta_cpu.float().mul_(mult)
                    else:
                        A_cpu, B_cpu, mult = entry[:3]
                        sl = entry[3] if len(entry) > 3 else None
                        se = entry[4] if len(entry) > 4 else None
                        if sl is not None and se is not None and B_cpu.shape[0] != (se - sl):
                            B_cpu = B_cpu[sl:se].contiguous()
                        delta_c = (B_cpu.float() @ A_cpu.float()).mul_(mult)
                    if sl is not None and se is not None:
                        target_rows = se - sl
                    else:
                        target_rows = w_new.shape[0]
                    delta_t = delta_c.to(target_dtype)
                    if delta_t.shape[0] == target_rows:
                        if sl is not None and se is not None:
                            w_new[sl:se] += delta_t
                        else:
                            w_new += delta_t
                    elif delta_t.shape[0] % target_rows == 0:
                        n = delta_t.shape[0] // target_rows
                        base_sl = sl if sl is not None else 0
                        for i in range(n):
                            seg = slice(base_sl + i * target_rows,
                                        base_sl + (i + 1) * target_rows)
                            w_new[seg] += delta_t[i * target_rows:(i + 1) * target_rows]
                    else:
                        log.warning(f"[int4 LoRA] shape mismatch delta={tuple(delta_t.shape)} "
                                    f"target={tuple(module.weight.shape)} — skip")
                        continue
                    applied.setdefault(_lname, []).append((delta_c.to(torch.float16).clone(), sl, se))
                module.weight = nn.Parameter(w_new)
                object.__setattr__(module.weight, "_model_dtype", target_dtype)
                if hasattr(module, "weight_comfy_model_dtype"):
                    module.weight_comfy_model_dtype = target_dtype
                if hasattr(module, "weight_comfyn"):
                    module.weight_comfyn = target_dtype
            else:
                # ── 原路径（无 AIMDO）：GPU 克隆 + GEMM ──
                w_new = module.weight.detach().clone()
                for _lname, entry in pairs:
                    if len(entry) >= 3 and isinstance(entry[0], str) and entry[0] == "delta":
                        _, delta_cpu, mult = entry[:3]
                        sl = entry[3] if len(entry) > 3 else None
                        se = entry[4] if len(entry) > 4 else None
                        delta_gpu = delta_cpu.to(device=w_dev, dtype=w_dtype).mul_(mult)
                    else:
                        A_cpu, B_cpu, mult = entry[:3]
                        sl = entry[3] if len(entry) > 3 else None
                        se = entry[4] if len(entry) > 4 else None
                        if sl is not None and se is not None and B_cpu.shape[0] != (se - sl):
                            B_cpu = B_cpu[sl:se].contiguous()
                        A_gpu = A_cpu.to(device=w_dev, dtype=w_dtype)
                        B_gpu = B_cpu.to(device=w_dev, dtype=w_dtype)
                        delta_gpu = (B_gpu @ A_gpu).mul_(mult)
                    if sl is not None and se is not None:
                        target_rows = se - sl
                    else:
                        target_rows = w_new.shape[0]
                    if delta_gpu.shape[0] == target_rows:
                        if sl is not None and se is not None:
                            w_new[sl:se] += delta_gpu
                        else:
                            w_new += delta_gpu
                    elif delta_gpu.shape[0] % target_rows == 0:
                        n = delta_gpu.shape[0] // target_rows
                        base_sl = sl if sl is not None else 0
                        for i in range(n):
                            seg = slice(base_sl + i * target_rows,
                                        base_sl + (i + 1) * target_rows)
                            w_new[seg] += delta_gpu[i * target_rows:(i + 1) * target_rows]
                    else:
                        log.warning(f"[int4 LoRA] shape mismatch delta={tuple(delta_gpu.shape)} "
                                    f"target={tuple(module.weight.shape)} — skip")
                        continue
                    applied.setdefault(_lname, []).append((delta_gpu.to(device=cpu, dtype=torch.float16).clone(), sl, se))
                module.weight = nn.Parameter(w_new)
        except Exception as e:
            log.warning("[int4 LoRA] bake pre-hook failed: %s", e)
        bs.pop('_pending', None)
        _prev_applied = bs.get('_applied')
        if isinstance(_prev_applied, list):
            _prev_applied = {"": _prev_applied}
        elif not isinstance(_prev_applied, dict):
            _prev_applied = {}
        for _ln, _lst in applied.items():
            _prev_applied[_ln] = _lst
        bs['_applied'] = _prev_applied
        bs.pop('_bake_now', None)
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
        # 节点执行 = 本次运行的真实意图：先丢弃卸载前登记的重放，避免旧 LoRA
        # 在新的一次运行里被自动装回
        _wa4_lora_replay_drop(model)
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
        _rec = next((
            x for x in _prev
            if isinstance(x, dict)
            and x.get("name") == lora_name
            and x.get("path") == lora_path
            and abs(float(x.get("strength", 1.0)) - float(strength)) < 1e-5
        ), None)
        if _rec is not None:
            if _rec.get("layers") == 0:
                # 上一次就 0 层匹配（LoRA 不属于这个模型）→ 不必每次重读文件重试
                if _WA4_REPLAY_CTX is None:
                    log.info("[int4 LoRA] = %s 上次匹配 0 层（模型里没有对应层），跳过重复尝试",
                             lora_name)
                return (model,)
            if _wa4_lora_state_live(model, lora_name):
                if _WA4_REPLAY_CTX is None:
                    log.info("[int4 LoRA] = %s 已在模型里（strength=%.2f），跳过重复注入",
                             lora_name, strength)
                return (model,)
            log.info("[int4 LoRA] %s 状态已丢失（去重记录仍在）→ 重新注入", lora_name)

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

        t0 = time.perf_counter()
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        fmt = _auto_detect_format(lora_sd)
        if fmt == "bfl": lora_sd = _convert_bfl_to_standard(lora_sd)
        lora_data = _parse_raw_lora_sd(lora_sd)

        if not getattr(model.model, '_wa4_detach_patched', False):
            # █ 原样保留（detach 清缓存逻辑，不动）█
            _orig_detach = model.detach
            def _wa4_detach(unpatch_all=True):
                _wa4_reset_all_loras(model, schedule_reapply=True)
                object.__setattr__(model.model, '_wa4_lora_needs_reset', True)
                return _orig_detach(unpatch_all)
            object.__setattr__(model, 'detach', _wa4_detach)
            object.__setattr__(model.model, '_wa4_detach_patched', True)

        aq, ab, unmatched = 0, 0, 0
        for norm, info in lora_data.items():
            lora_type = info.get("type", "standard")
            is_qkv = (norm.endswith(".attn.qkv") or norm.endswith(".attn1.qkv")
                      or norm.endswith(".attn2.qkv")
                      or norm.endswith(".attn.wq") or norm.endswith(".attn.wk")
                      or norm.endswith(".attn.wv"))
            if is_qkv:
                if norm.endswith(".attn.qkv"):
                    targets = _resolve_qkv_slices(index, norm)
                else:
                    # to_q/to_k/to_v（归一化为 wq/wk/wv）：模型 fused qkv 时
                    # 取对应段（q:0..hs / k:hs..2hs / v:2hs..3hs），否则全量。
                    seg = {"wq": 0, "wk": 1, "wv": 2}[norm.rsplit(".", 1)[-1]]
                    base = norm.rsplit(".attn", 1)[0]
                    qkv_mods = _resolve_with_alias(index, base + ".attn.qkv")
                    if qkv_mods:
                        m = qkv_mods[0]
                        out_f = (m.out_features if hasattr(m, "out_features")
                                 else (m.weight.shape[0] if hasattr(m, "weight")
                                       and m.weight is not None else 0))
                        if out_f > 0 and out_f % 3 == 0:
                            hs = out_f // 3
                            targets = [(norm, (seg * hs, (seg + 1) * hs))]
                        else:
                            targets = [(norm, None)]
                    else:
                        targets = [(norm, None)]
            else:
                targets = [(norm, None)]
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
        if aq: parts.append(f"{aq} quant")
        if ab: parts.append(f"{ab} bake")
        if aq == 0 and ab == 0: parts = ["0 layers"]
        if _WA4_REPLAY_CTX is None:
            log.info("[int4 LoRA] ✓ 注入 %s | %s | strength=%s | %.2fs%s",
                     lora_name, " + ".join(parts), strength, elapsed,
                     f" | {unmatched} unmatched" if unmatched else "")
        else:
            _WA4_REPLAY_CTX["q"] += aq
            _WA4_REPLAY_CTX["b"] += ab
            log.debug("[int4 LoRA] （自动补回）注入 %s | %s | strength=%s | %.2fs",
                      lora_name, " + ".join(parts), strength, elapsed)

        if not hasattr(model.model, '_wa4_loras'):
            object.__setattr__(model.model, '_wa4_loras', [])
        model.model._wa4_loras.append({"name": lora_name, "strength": strength,
                                       "path": lora_path, "layers": aq + ab})
        del lora_sd, lora_data
        return (model,)

    @staticmethod
    def _pop_module_lora(module, lora_name):
        # 移除该 LoRA 在本层的一切痕迹：量化条目 + 它自己的 baked delta。
        # 关键：同一层上其它 LoRA 的 delta/hook **必须保留**（以前是一刀切
        # 回滚整层 _applied，多 LoRA 同层时会把别人的 bake 一起清掉）。
        # 返回值 (量化条目数, bake 层数) 供日志统计。
        had_q = had_b = 0
        if _is_quant_linear(module):
            le = getattr(module, '_wa4_lora_entries', None)
            if le is not None:
                if le.pop(lora_name, None) is not None:
                    had_q = 1
                if len(le) == 0: object.__setattr__(module, '_wa4_lora_entries', None)
        bs = getattr(module, '_wa4_bake_state', None)
        if bs is None: return had_q, had_b
        pend = bs.get('_pending')
        if isinstance(pend, dict):
            if pend.pop(lora_name, None) is not None:
                had_b = 1      # 排队中就被清理掉的 bake 也要计数（口径：本层被清了什么）
        app = bs.get('_applied')
        deltas = None
        if isinstance(app, dict):
            deltas = app.pop(lora_name, None)
        elif app:
            # 兼容旧结构：整层一份，只能整体回滚
            deltas = app
            app = None
        if deltas and hasattr(module, 'weight') and module.weight is not None:
            had_b = 1
            for delta_cpu, sl, se in deltas:
                try:
                    neg = (-delta_cpu).to(device=module.weight.device, dtype=module.weight.dtype)
                    if sl is not None and se is not None:
                        module.weight.data[sl:se].add_(neg)
                    else:
                        module.weight.data.add_(neg)
                except Exception: pass
        # 本层还有其它 LoRA（排队中或已 baked）→ 保留 hook 与状态；彻底空了才清理
        if not pend and not app:
            bs.pop(lora_name, None)
            bs.pop('_pending', None)
            bs.pop('_applied', None)
            bs.pop('_bake_now', None)
            hh = bs.pop('_hook_handle', None)
            if hh is not None:
                try: hh.remove()
                except Exception: pass
            bs.clear()
        return had_q, had_b

    def _remove_lora(self, model, lora_name):
        # █ 原样保留 █
        # 额外 1：同步撤掉去重记录，否则再次加载同名 LoRA 会被误判为"已注入"。
        # 额外 2：把实际清掉的东西打进日志（以前这条路径完全没有痕迹，
        # 出问题时无法从日志判断 strength=0 到底有没有生效）。
        bm = model.model
        while hasattr(bm, '_orig_mod'): bm = bm._orig_mod
        nq = nb = 0
        for m in bm.modules():
            q, b = self._pop_module_lora(m, lora_name)
            nq += q; nb += b
        prev = getattr(model.model, '_wa4_loras', None) or []
        keep = [x for x in prev
                if not (isinstance(x, dict) and x.get("name") == lora_name)]
        object.__setattr__(model.model, '_wa4_loras', keep)
        if nq or nb or len(keep) != len(prev):
            log.info("[int4 LoRA] ✗ 移除 %s（strength=0）：清理 %d 个量化层条目 + %d 个 bake 层",
                     lora_name, nq, nb)
        else:
            log.debug("[int4 LoRA] %s 本来就不在模型里（strength=0，空操作）", lora_name)

    def _inject_standard(self, module, lora_name, down, up, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False):
        # █ 原样保留 █
        A = down.to(cpu, torch.float16).clone()
        B = up.to(cpu, torch.float16).clone()
        # 形状预检：LoRA out/in 必须匹配模块，否则在加载时一次性过滤，
        # 避免条目进入 forward 热路径后每次白算矩阵乘再跳过刷屏。
        if qkv_slice is not None:
            target_out = qkv_slice[1] - qkv_slice[0]
        else:
            target_out = (module.out_features if hasattr(module, "out_features")
                          else (module.weight.shape[0]
                                if module.weight is not None else None))
        target_in = (module.in_features if hasattr(module, "in_features")
                     else (module.weight.shape[1]
                           if module.weight is not None else None))
        if (target_out is not None and B.shape[0] != target_out) or (
                target_in is not None and A.shape[1] != target_in):
            log.warning(
                "[int4 LoRA] shape mismatch: LoRA=(out %s, in %s) vs "
                "module=(out %s, in %s) (%s) — skip",
                B.shape[0], A.shape[1], target_out, target_in, lora_name)
            return
        if quarot_enabled and H is not None: A = _rot_quarot_tensor(A, H, group_size)
        rank = up.shape[1] if up.ndim >= 2 else 1
        mult = ((alpha_val / max(rank, 1)) if alpha_val else 1.0) * strength
        if bake:
            bs = getattr(module, '_wa4_bake_state', None)
            if bs is None: bs = {}; object.__setattr__(module, '_wa4_bake_state', bs)
            pending = bs.get('_pending')
            if not isinstance(pending, dict): pending = {}; bs['_pending'] = pending
            sl = qkv_slice[0] if qkv_slice else None
            se = qkv_slice[1] if qkv_slice else None
            # 按 LoRA 名字分别排队：同层多 LoRA 时各自的 delta 独立记账
            pending.setdefault(lora_name, []).append((A, B, mult, sl, se))
            if '_hook_handle' not in bs:
                _bake_fn = _make_bake_pre_hook(module)
                hook = module.register_forward_pre_hook(_bake_fn)
                bs['_hook_handle'] = hook
                bs['_bake_now'] = _bake_fn   # 卸载重放时立即补齐（避免早于 int4 层的 baked 层漏掉首个 step）
        else:
            le = getattr(module, '_wa4_lora_entries', None)
            if le is None: le = {}; object.__setattr__(module, '_wa4_lora_entries', le)
            if qkv_slice is not None:
                sl, se = qkv_slice
                if B.shape[0] != (se - sl):
                    # LoRA 的 B 是全量 fused qkv（out == 段总宽）→ 按段切
                    B_seg = B[sl:se].contiguous().clone()
                else:
                    # LoRA 的 B 本身就是单段（to_q/to_k/to_v，out == 段宽）
                    # → 不切，全量注入到 o 的 sl:se
                    B_seg = B.contiguous().clone()
                le.setdefault(lora_name, []).append(
                    (A, B_seg, mult, sl, se))
            else:
                le.setdefault(lora_name, []).append((A, B, mult))

    def _inject_lokr(self, module, lora_name, w1, w2, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False):
        # █ 原样保留 █
        w1_c = w1.to(cpu, torch.float16).clone(); w2_c = w2.to(cpu, torch.float16).clone()
        to2 = module.out_features if hasattr(module, "out_features") else module.weight.shape[0]
        ti2 = module.in_features if hasattr(module, "in_features") else module.weight.shape[1]
        # ── LoKR 因子快速路径：不物化 kron(delta) ──────────────────────────
        # 仅当 kron 行数==out 且输入宽正好是若干整段 w2 列（realism/Krea2 模式）
        # 时使用；其余情形（bake/qkv 分片/quarot/不整除）原样走下方旧路径。
        r1, c1 = w1_c.shape
        r2, c2 = w2_c.shape
        if (not bake and qkv_slice is None and not quarot_enabled
                and r1 * r2 == to2 and ti2 % c2 == 0 and 0 < ti2 // c2 <= c1):
            le = getattr(module, '_wa4_lora_entries', None)
            if le is None:
                le = {}
                object.__setattr__(module, '_wa4_lora_entries', le)
            le.setdefault(lora_name, []).append(
                ("lokr", w1_c.contiguous(), w2_c.contiguous(), strength))
            return
        delta = torch.kron(w1_c, w2_c)
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
            if not isinstance(pending, dict): pending = {}; bs['_pending'] = pending
            sl = qkv_slice[0] if qkv_slice else None
            se = qkv_slice[1] if qkv_slice else None
            pending.setdefault(lora_name, []).append(("delta", delta, strength, sl, se))
            if '_hook_handle' not in bs:
                _bake_fn = _make_bake_pre_hook(module)
                hook = module.register_forward_pre_hook(_bake_fn)
                bs['_hook_handle'] = hook
                bs['_bake_now'] = _bake_fn   # 卸载重放时立即补齐（避免早于 int4 层的 baked 层漏掉首个 step）
        else:
            le = getattr(module, '_wa4_lora_entries', None)
            if le is None: le = {}; object.__setattr__(module, '_wa4_lora_entries', le)
            if qkv_slice is not None:
                sl, se = qkv_slice
                le.setdefault(lora_name, []).append(
                    ("delta", delta[sl:se, :].contiguous().clone(), strength, sl, se))
            else:
                le.setdefault(lora_name, []).append(("delta", delta, strength))


def _wa4_lora_replay_apply(model, specs):
    """模型卸载清空 LoRA 后，在模型重新运行时自动补回（仅同一次运行内）。

    走与节点完全相同的注入路径（含去重记录写入），状态与"节点刚执行过"等价；
    重读文件约 0.1~0.2s/LoRA。每 LoRA 的注入明细降到 DEBUG，结束时只打一条
    可读汇总（量化层数 / bake 层数 / 耗时 / 是否重启预热）。
    """
    global _WA4_REPLAY_CTX
    from .int4_xpu_loader import _wa4_arm_prewarm, _wa4_lora_short_names
    ctx = {"q": 0, "b": 0}
    _WA4_REPLAY_CTX = ctx
    t0 = time.perf_counter()
    names, n_bake, armed = [], 0, False
    try:
        loader = INT4XPULoRALoader()
        for name, strength in specs:
            try:
                loader.load_lora(model, name, strength)
                names.append(name)
            except Exception as e:
                log.warning("[int4 LoRA] 自动补回 %s 失败：%s", name, e)
        n_bake = _wa4_lora_replay_flush_bakes(model)
        armed = _wa4_arm_prewarm(model)
    finally:
        _WA4_REPLAY_CTX = None
    if names:
        log.info(
            "[int4 LoRA] 模型重新运行：自动重新注入 %d 个 LoRA"
            "（量化层 %d、bake 层 %d 已换权%s，%.2fs）：%s",
            len(names), ctx["q"], n_bake,
            "，预热已重启" if armed else "", time.perf_counter() - t0,
            _wa4_lora_short_names(names))


def _wa4_lora_replay_flush_bakes(model):
    """把重放刚排队的 baked delta 立即落盘到权重。

    正常路径下 bake 由各层 forward pre-hook 在"该层首次前向"时执行；重放发生
    在某层前向内部，早于首个 int4 层的 baked 层已经跑过本步前向，会漏掉首个
    step 的 delta（实测图像残差 mean≈0.7/255）。这里在重放结束时统一立即执行，
    使重放结果与"节点刚执行过"完全一致。返回实际换权的层数（调用方负责日志）。
    """
    bm = model.model
    while hasattr(bm, '_orig_mod'): bm = bm._orig_mod
    n = 0
    for m in bm.modules():
        bs = getattr(m, '_wa4_bake_state', None)
        if not bs or not bs.get('_pending'):
            continue
        fn = bs.get('_bake_now')
        if fn is None:
            continue
        try:
            fn(m, None)
            n += 1
        except Exception as e:
            log.warning("[int4 LoRA] 重放 bake 立即执行失败（将由 pre-hook 兜底）：%s", e)
    return n


NODE_CLASS_MAPPINGS = {"INT4XPULoRALoader": INT4XPULoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4XPULoRALoader": "INT4XPU LoRA Loader"}
