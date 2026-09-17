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
from . import int4_xpu_lora_sets as int4_lora_sets
from .int4_xpu_loader import _is_quant_linear
from .int4_xpu_lora_common import (
    _wa4_reset_all_loras,
    _auto_detect_format, _convert_bfl_to_standard,
    _parse_raw_lora_sd, _get_accelerator_device, _rot_quarot_tensor,
    _resolve_with_alias,
)

log = logging.getLogger("int4-LoRA")


def _wa4_parts(q, b):
    """日志里的层数描述：224 quant + 32 bake / 0 layers。"""
    if not q and not b:
        return "0 layers"
    out = []
    if q: out.append(f"{q} quant")
    if b: out.append(f"{b} bake")
    return " + ".join(out)


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


class _Wa4LoraWorker:
    """LoRA 注入的干活的：把一份 LoRA 应用到模型上 / 从模型上移除。

    不持有状态，不写日志结论（日志由 int4_xpu_lora_sets 按"采样阶段"统一打）。
    """

    def apply_lora(self, model, lora_name, strength, entry_key=None):
        """把 lora_name 按 strength 注入模型；返回 {quant, bake, unmatched}。

        entry_key：条目/记账用的唯一键（同一个 LoRA 在链里出现多次时区分各份），
        默认就是 lora_name。文件查找始终用 lora_name。
        """
        key = entry_key or lora_name
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"[int4 LoRA] '{lora_name}' not found")

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
        lora_data = int4_lora_sets.parse_lora(lora_path)

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
                        res = _resolve_lokr(info)
                        if res is None:
                            log.debug("[int4 LoRA] LoKr 因子无法解析，跳过 %s", norm)
                            continue
                        w1, w2, lokr_rank = res
                        self._inject_lokr(module, key, w1, w2, info.get("alpha"), strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=not is_quant, rank=lokr_rank)
                    else:
                        down = info.get("down"); up = info.get("up")
                        if down is None or up is None: continue
                        self._inject_standard(module, key, down, up, info.get("alpha"), strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=not is_quant)
                    if is_quant: aq += 1
                    else: ab += 1
                    layer_matched = True
            if not layer_matched: unmatched += 1

        elapsed = time.perf_counter() - t0
        log.debug("[int4 LoRA] 注入完成 %s | %s | strength=%s | %.2fs%s",
                  lora_name, _wa4_parts(aq, ab), strength, elapsed,
                  f" | {unmatched} unmatched" if unmatched else "")
        del lora_data
        return {"quant": aq, "bake": ab, "unmatched": unmatched}

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

    def remove_lora(self, model, lora_name):
        """把该 LoRA 从模型上彻底移除（只动它自己的条目与 baked delta）。"""
        bm = model.model
        while hasattr(bm, '_orig_mod'): bm = bm._orig_mod
        nq = nb = 0
        for m in bm.modules():
            q, b = self._pop_module_lora(m, lora_name)
            nq += q; nb += b
        log.debug("[int4 LoRA] 移除 %s：%d 个量化层条目 + %d 个 bake 层", lora_name, nq, nb)

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

    def _inject_lokr(self, module, lora_name, w1, w2, alpha_val, strength, qkv_slice, quarot_enabled, H, group_size, dev, cpu, bake=False, rank=None):
        # w2 可能是 (w2_a, w2_b) 因子对（LyCORIS 原生分解写法，w2 = a @ b）：
        # 量化前向里按 a/b 直接做两级小 GEMM，不合并、不物化 kron。
        split = isinstance(w2, (tuple, list)) and len(w2) == 2
        w2a, w2b = (w2 if split else (None, None))
        # 原生 LoKr 语义：alpha 只在因子做了 a/b 分解时生效（乘 alpha/rank）；
        # 直给 w1/w2 时 alpha 被原生忽略（社区文件里的 alpha 常是垃圾值）。
        mult = float(strength)
        if alpha_val is not None and rank:
            try:
                mult *= float(alpha_val) / float(rank)
            except Exception:
                pass
        to2 = module.out_features if hasattr(module, "out_features") else module.weight.shape[0]
        ti2 = module.in_features if hasattr(module, "in_features") else module.weight.shape[1]
        # ── 量化层：只存因子，前向按因子做小 GEMM（不物化 kron(delta)）─────
        # 旧实现"每层前向临时物化一份 kron delta"，大层上就是每层几百 MB 的
        # 瞬时分配（qwen LoKr 840 层直接撑爆 15.4GB 上限）；这里彻底去掉。
        if not bake and qkv_slice is None and not quarot_enabled:
            le = getattr(module, '_wa4_lora_entries', None)
            if le is None:
                le = {}
                object.__setattr__(module, '_wa4_lora_entries', le)
            w1_c = w1.to(cpu, torch.float16).contiguous().clone()
            if split:
                a_c = w2a.to(cpu, torch.float16).contiguous().clone()
                b_c = w2b.to(cpu, torch.float16).contiguous().clone()
                le.setdefault(lora_name, []).append(("lokr2", w1_c, a_c, b_c, mult))
            else:
                w2_c = w2.to(cpu, torch.float16).contiguous().clone()
                le.setdefault(lora_name, []).append(("lokr", w1_c, w2_c, mult))
            return
        # ── 其余（bake / fused qkv 分片 / quarot 旋转）：需要具体的 delta ──
        w1_c = w1.to(cpu, torch.float16).clone()
        if split:
            w2_c = (w2a.to(cpu, torch.float16) @ w2b.to(cpu, torch.float16)).clone()
        else:
            w2_c = w2.to(cpu, torch.float16).clone()
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
            pending.setdefault(lora_name, []).append(("delta", delta, mult, sl, se))
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
                    ("delta", delta[sl:se, :].contiguous().clone(), mult, sl, se))
            else:
                le.setdefault(lora_name, []).append(("delta", delta, mult))


def _resolve_lokr(info):
    """解析 LoKr 因子：LyCORIS 的 a/b 分解与 tucker 形式统一成注入用形式。

    返回 (w1, w2, rank)：w2 是张量（直因子）或 (w2_a, w2_b) 因子对；
    rank 是原生 alpha/dim 里的 dim（只有用了 a/b 分解时非 None）。
    返回 None 表示不支持（例如 tucker 卷积的 4D 因子）。
    """
    cached = info.get("_lokr_res", False)
    if cached is not False:
        return cached
    w1, w1a, w1b = info.get("lokr_w1"), info.get("lokr_w1_a"), info.get("lokr_w1_b")
    w2, w2a, w2b = info.get("lokr_w2"), info.get("lokr_w2_a"), info.get("lokr_w2_b")
    t2 = info.get("lokr_t2")
    rank = None
    ok = True
    if w1 is None:
        if w1a is not None and w1b is not None:
            w1 = w1a.to(torch.float32) @ w1b.to(torch.float32)   # 外因子体积小，直接合并
            rank = int(w1b.shape[0])
        else:
            ok = False
    if ok and w2 is None:
        if w2a is not None and w2b is not None:
            if t2 is None:
                w2 = (w2a, w2b)                                  # 内因子保持 a/b，前向两级小 GEMM
            else:
                w2 = torch.einsum("i j k l, j r, i p -> p r k l",
                                  t2.to(torch.float32), w2b.to(torch.float32), w2a.to(torch.float32))
                if w2.dim() == 4 and w2.shape[-1] == 1 and w2.shape[-2] == 1:
                    w2 = w2.squeeze(-1).squeeze(-1)
                if w2.dim() != 2:
                    ok = False                                   # 卷积 tucker：本插件不处理
            if ok and rank is None:
                rank = int(w2b.shape[0])
        else:
            ok = False
    res = (w1, w2, rank) if (ok and w1 is not None and w2 is not None) else None
    info["_lokr_res"] = res
    return res


_WORKER = _Wa4LoraWorker()


def parse_lora_file(path):
    """读 LoRA 文件 → {层路径: {type, down/up/alpha 或 lokr_w1/w2}}（由规格层缓存）。"""
    lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
    fmt = _auto_detect_format(lora_sd)
    if fmt == "bfl":
        lora_sd = _convert_bfl_to_standard(lora_sd)
    return _parse_raw_lora_sd(lora_sd)


def apply_lora(model, lora_name, strength, entry_key=None):
    return _WORKER.apply_lora(model, lora_name, strength, entry_key=entry_key)


def remove_lora(model, lora_name):
    return _WORKER.remove_lora(model, lora_name)


def install_detach(model):
    """卸载包装：清空 LoRA 状态（回滚 bake）+ 标记失效；下次采样按活跃规格重建。

    幂等标志挂在 **patcher 实例** 上 —— 节点 clone 后每个分支各装一份。
    """
    if getattr(model, "_wa4_lora_detach_patched", False):
        return
    _orig_detach = model.detach

    def _wa4_detach(unpatch_all=True):
        try:
            from .int4_xpu_loader import _wa4_release_model
            _wa4_release_model(model)
        except Exception as e:
            log.debug("[int4 LoRA] 显存释放失败：%s", e)
        try:
            _wa4_reset_all_loras(model)
            int4_lora_sets.mark_dirty(model)
        except Exception as e:
            log.debug("[int4 LoRA] 卸载清理失败：%s", e)
        return _orig_detach(unpatch_all)

    object.__setattr__(model, 'detach', _wa4_detach)
    object.__setattr__(model, '_wa4_lora_detach_patched', True)


def _is_int4_model(model) -> bool:
    """判断这个 MODEL 是否由本插件的 int4 加载器加载。

    判定与采样侧对齐钩子一致：patcher 上带 `wa4_int4`，或 diffusion_model 上带
    `_wa4_lora_index`（原生加载器加载的模型两个都没有）。
    """
    try:
        if (getattr(model, "model_options", None) or {}).get("wa4_int4"):
            return True
    except Exception:
        pass
    dm = getattr(model, "model", None)
    while hasattr(dm, "_orig_mod"):
        dm = dm._orig_mod
    return dm is not None and getattr(dm, "_wa4_lora_index", None) is not None


def _native_load_lora_model_only(model, lora_name, strength):
    """非 int4 模型：完全走 ComfyUI 原生 LoRA 路径（与原生 LoraLoaderModelOnly 同源）。"""
    try:
        from nodes import LoraLoaderModelOnly
        return LoraLoaderModelOnly().load_lora_model_only(model, lora_name, float(strength))
    except Exception as e:
        log.warning("[int4 LoRA] 原生模型走原生 LoRA 路径失败（%s），本节点跳过：%s",
                    lora_name, e)
        return (model,)


class INT4XPULoRALoader:
    """薄壳节点：只记录"这一路要哪些 LoRA"，实际应用在采样时按活跃规格对齐。

    与原生 LoRA 节点同语义：`model.clone()` 后再追加规格 —— chained 节点叠加、
    并联分支各自独立、绕过节点等于不在链里。

    非 int4 模型（原生加载器加载的 bf16/fp8/int8-convrot 等）：直接委托给
    ComfyUI 原生 `LoraLoaderModelOnly`，行为与原生节点完全一致（含原生支持的
    LoHa/LoKr/OFT/DoRA 等格式与 strength=0 直通）。
    """

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

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    DESCRIPTION = "在模型上叠加一个 LoRA（采样时应用，语义与原生 LoRA 节点一致）。"

    def load_lora(self, model, lora_name, strength):
        if not _is_int4_model(model):
            return _native_load_lora_model_only(model, lora_name, strength)
        parent = model
        model = model.clone()
        install_detach(model)
        # strength=0 也要登记规格：链里前面挂过同名 LoRA 时执行"归零=撤销"
        # （int4_xpu_lora_sets._apply_zero_cancel，插件既有能力）；
        # 链里没有同名时它就是纯 no-op（与原生一致：本次采样不注入）。
        int4_lora_sets.register(parent, model, [
            {"name": lora_name, "strength": float(strength), "kind": "loader"}])
        return (model,)


NODE_CLASS_MAPPINGS = {"INT4XPULoRALoader": INT4XPULoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4XPULoRALoader": "INT4XPU LoRA Loader"}
