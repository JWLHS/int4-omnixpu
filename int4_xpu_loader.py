"""
INT4XPU Model Loader v1.4.2n — omni_xpu_kernel.svdq oneDNN INT4 GEMM via XMX engine.
Pre-injection + FP8转换 + dtype 按文件标记 + omni norm V3 + QuaRot + LoRA + 显存释放。

v1.4.2n（实验A）：探针无条件安装（AIMDO active 也装，原版 reset+sync 机制）；
  其余（patch_aimdo_xpu / Krea2 常驻 / detach 释放）让路逻辑全保留。
v1.4.3（AIMDO 全权接管）：
  - 移除 QW attention 探针（reset_peak+sync，不再强制同步）。
  - 显存策略全部按 aimdo_active() 让路：AIMDO 接管时 wa4 不做
    release_xpu / synchronize / empty_cache；未启用 AIMDO 时保留原兜底。
  - 保留设备放置（forward 惰性 .to(dev)，AIMDO 管不到）；
    release_xpu 补上 _w_s8。
v1.4.2m:
  - + QW attention 探针（'Attention' 子串匹配 + reset/sync/peak）——实证 12.5GB 不溢出
  - + Krea2 常驻 GPU + AIMDO 让路
"""
import gc, os, time, types, torch, torch.nn as nn, torch.nn.functional as F, comfy.ops, comfy.utils
import comfy.model_detection, comfy.sd, folder_paths, logging

log = logging.getLogger("wa4")

_FP8_TYPES = set()
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _t = getattr(torch, _name, None)
    if _t is not None: _FP8_TYPES.add(_t)

_OMNI_NORM_SKIP = {"Boogu", "QwenImage", "Wan", "CogVideoX", "ZImage", "Lens", "Lightricks"}
_DETACH_RELEASE_MODELS = {"QwenImage"}

_WA4_SYNC = os.environ.get("OMNIXPU_INT4_SYNC", "1") != "0"
_WA4_SYNC_EVERY = int(os.environ.get("OMNIXPU_INT4_SYNC_EVERY", "64"))
_RAM_TRACE = os.environ.get("OMNIXPU_RAM_TRACE", "0") != "0"
_AIMDO_EMPTY = os.environ.get("OMNIXPU_INT4_AIMDO_EMPTY", "0") != "0"
_AIMDO_EMPTY_TRACE = os.environ.get("OMNIXPU_INT4_EMPTY_TRACE", "0") != "0"
_TIMING = os.environ.get("OMNIXPU_INT4_TIMING", "0") != "0"
_TINT4_NATIVE = os.environ.get("OMNIXPU_INT4_TINT4_NATIVE", "1") != "0"
_LOAD_T0 = 0.0


def _kernel_caps():
    """omni_xpu_kernel 可用算子检测（kernel 分 A/B 版、更新频繁，加载时探测）。"""
    caps = {"w4a4": False, "s8u4": False, "int4": False, "tint4": False}
    try:
        from omni_xpu_kernel import svdq
        # w4a4 预留：kernel 已有 w4a4_gemm_esimd，但层间 INT4 激活传递未实现，
        # 达成前保持 False，加载时自动逐级回退 w4a8 → w4a16。
        caps["w4a4"] = False
        caps["s8u4"] = hasattr(svdq, "onednn_s8u4_gemm")
        caps["int4"] = (
            hasattr(svdq, "onednn_int4_gemm_preconverted")
            or hasattr(svdq, "onednn_int4_gemm")
        )
        # tint4 原生：A 系列 fork 有独立 onednn_int4_gemm_tint4；
        # B 系列（原仓库 PR）是 onednn_int4_gemm_preconverted 带 zp 参数。
        caps["tint4"] = (
            hasattr(svdq, "onednn_int4_gemm_tint4")
            or hasattr(svdq, "onednn_int4_gemm_preconverted")
        )
        caps["tint4_native_op"] = (
            hasattr(svdq, "onednn_int4_gemm_tint4")
            or hasattr(svdq, "onednn_int4_gemm_torchao")
        )
    except Exception:
        pass
    return caps


_BACKEND_CHAIN = {
    "w4a4": ("w4a4", "w4a8", "w4a16"),
    "w4a8": ("w4a8", "w4a16"),
    "w4a16": ("w4a16",),
}


def _resolve_backend(requested, caps):
    """逐级回退：w4a4 → w4a8 → w4a16（kernel 缺失时逐级降级）。
    w4a16 恒可达（wa4 走纯 python / tint4 走 torchao 兜底）。
    未来 w4a4 层间 INT4 传递实现后，把 caps['w4a4'] 置 True 即自动启用。"""
    req = requested if requested in _BACKEND_CHAIN else "w4a16"
    for cand in _BACKEND_CHAIN[req]:
        if cand == "w4a4":
            if caps.get("w4a4"):
                return cand
            log.warning("[int4] w4a4 backend 未就绪（层间 INT4 激活传递未实现），逐级回退 w4a8")
            continue
        if cand == "w4a8":
            if caps["s8u4"]:
                return cand
            log.warning("[int4] a8 kernel 不可用（onednn_s8u4_gemm 缺失），"
                        "自动回退 w4a16")
            continue
        if cand == "w4a16":
            return cand
    return "w4a16"


class Int4LinearPython(nn.Module):
    """wa4 格式在无 omnixpu kernel 时的纯 python 回退：逐层反量化 + F.linear。

    注意：反量化结果不跨 forward 缓存（用完即弃），否则 224 层 fp16 权重
    全部驻留会把 8GB+ 的 int4 模型撑爆显存（OOM）。回退本来就慢，优先保内存。
    """

    def __init__(self, in_features, out_features, packed, scale, group_size,
                 bias=None, act_dtype=torch.float16, use_quarot=False,
                 hadamard_H=None, correction=None, quarot_gs=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        object.__setattr__(self, "_packed", packed)  # [N, K/2] u8（有符号 nibble）
        object.__setattr__(self, "_scale", scale)     # [G, N] f16
        object.__setattr__(self, "_gs", group_size)
        object.__setattr__(self, "_correction", correction)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._act_dtype = act_dtype
        self._use_quarot = use_quarot
        self._hadamard_H = hadamard_H
        self._quarot_gs = quarot_gs or group_size
        object.__setattr__(self, "_wa4_lora_entries", None)

    def _dequant(self, dev):
        packed = self._packed.to(dev)
        N, half = packed.shape
        K = half * 2
        b = packed.view(torch.uint8)
        lo = (b & 0x0F).to(torch.int32)
        hi = ((b >> 4) & 0x0F).to(torch.int32)
        q = torch.stack([lo, hi], dim=-1).reshape(N, K)
        qs = torch.where(q >= 8, q - 16, q).float()
        sc = self._scale.to(dev).float().repeat_interleave(self._gs, dim=0).t()
        w = qs * sc
        if self._correction is not None:
            w = w + self._correction.to(dev).float().t()
        return w.to(self._act_dtype)

    def forward(self, x):
        dev = x.device
        x2 = x.reshape(-1, x.shape[-1]).to(self._act_dtype)
        if self._use_quarot and self._hadamard_H is not None:
            try:
                from .int4_xpu_quarot import rotate_activation
                x2 = rotate_activation(x2, self._hadamard_H, self._quarot_gs)
            except Exception:
                pass
        w = self._dequant(dev)  # 局部反量化，forward 结束即释放
        out = F.linear(x2, w)
        del w
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=out.dtype)
        return out.reshape(*x.shape[:-1], out.shape[-1])


class Int4LinearTorchao(nn.Module):
    """tint4 格式在无 omnixpu kernel 时的 torchao 回退（与 tint4 原厂同路径）。"""

    def __init__(self, in_features, out_features, qdata, zp, scale, group_size,
                 bias=None, act_dtype=torch.float16, use_quarot=False,
                 hadamard_H=None, quarot_gs=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        object.__setattr__(self, "_qdata", qdata)
        object.__setattr__(self, "_zp", zp)
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_gs", group_size)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._act_dtype = act_dtype
        self._use_quarot = use_quarot
        self._hadamard_H = hadamard_H
        self._quarot_gs = quarot_gs or group_size
        object.__setattr__(self, "_qt", None)
        object.__setattr__(self, "_wa4_lora_entries", None)

    def forward(self, x):
        dev = x.device
        x2 = x.reshape(-1, x.shape[-1])
        if self._use_quarot and self._hadamard_H is not None:
            try:
                from .int4_xpu_quarot import rotate_activation
                x2 = rotate_activation(x2, self._hadamard_H, self._quarot_gs)
            except Exception:
                pass
        try:
            if self._qt is None or getattr(self._qt, "device", None) != dev:
                from torchao.quantization.quantize_.workflows.int4.int4_plain_int32_tensor import (
                    Int4PlainInt32Tensor,
                )
                self._qt = Int4PlainInt32Tensor(
                    self._qdata.to(dev), self._scale.to(dev),
                    self._zp.to(device=dev, dtype=torch.int8),
                    [1, self._gs], [self.out_features, self.in_features],
                )
            out = F.linear(x2, self._qt, None)
        except Exception as _tao_e:
            # torchao 缺失/过旧时的纯 python 回退：
            # 解包 nibbles -> w = (q - zp) * scale -> F.linear。
            w = self._python_dequant(dev)
            out = F.linear(x2, w)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=out.dtype)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def _python_dequant(self, dev):
        """tint4 int32 qdata -> [N, K] 权重 = (q - zp) * scale（纯 python）。"""
        qdata = self._qdata.to(dev)
        N, K8 = qdata.shape
        K = K8 * 8
        b = qdata.view(torch.uint8)  # [N, K/2]，小端 2 nibble/byte
        lo = (b & 0x0F).to(torch.int32)
        hi = ((b >> 4) & 0x0F).to(torch.int32)
        q = torch.stack([lo, hi], dim=-1).reshape(N, K)  # [N, K] u4
        G = self._zp.shape[0]
        gs = self._gs
        z = self._zp.to(dev).float().t().unsqueeze(-1).expand(N, G, gs).reshape(N, K)
        s = self._scale.to(dev).float().t().unsqueeze(-1).expand(N, G, gs).reshape(N, K)
        return ((q - z) * s).to(self._act_dtype)


def _tphase(name, mark=None):
    """分阶段计时（仅 OMNIXPU_INT4_TIMING=1 时输出，默认零开销）。"""
    global _LOAD_T0
    if not _TIMING:
        return time.time()
    now = time.time()
    if _LOAD_T0 == 0.0:
        _LOAD_T0 = now
    if mark is None:
        mark = _LOAD_T0
    log.info("[wa4-timing] %s: %.2fs (total %.2fs)", name, now - mark, now - _LOAD_T0)
    return now


def _sync_point():
    """Shared periodic sync + optional AIMDO pool reclaim.

    oneDNN GEMMs execute asynchronously on the torch queue; without a
    periodic host-side sync their USM allocations and the driver-level
    footprint stay pinned high. AIMDO's physical page pool also keeps pages
    after the run (physical_release ~= 0), so when OMNIXPU_INT4_AIMDO_EMPTY=1
    we additionally ask AIMDO to release cached pages at the same point.
    """
    if not _WA4_SYNC:
        return
    if INT4XPULinear._call_count % _WA4_SYNC_EVERY == 0:
        torch.xpu.synchronize()
        if _AIMDO_EMPTY:
            try:
                from comfy_aimdo import control as _aimdo_ctl
                _ok = _aimdo_ctl.empty_xpu_allocator_cache(wait=True)
                if _AIMDO_EMPTY_TRACE:
                    log.info(
                        "[int4] aimdo empty -> %s total=%.0fMB",
                        _ok,
                        _aimdo_ctl.get_total_vram_usage() / (1024**2),
                    )
            except Exception as _e:
                if _AIMDO_EMPTY_TRACE:
                    log.warning("[int4] aimdo empty failed: %r", _e)


def _ram_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return -1.0


def _ram_trace(tag):
    if _RAM_TRACE:
        log.info("[int4][ram] %-28s %.2f GB", tag, _ram_gb())


def _empty_cache_enabled():
    return os.environ.get("OMNIXPU_INT4_EMPTY_CACHE", "0") != "0"


def _empty_cache():
    try:
        torch.xpu.empty_cache()
    except Exception:
        pass


class W4ActS8:
    """Layer-to-layer s8 activation for the w4a8 closure (A8_LAYER_CLOSURE.md).

    Carries the quantized activation (a8 [M,K] int8) and its per-group scale
    ([M, K/gs] f16) together with the original tensor shape/dtype, so a
    INT4XPULinear can consume it directly (skipping activation quantization) or
    dequantize at residual/nonlinear/attention boundaries.
    """

    __slots__ = ("a8", "scale", "orig_shape", "dtype", "gs")

    def __init__(self, a8, scale, orig_shape, dtype, gs):
        self.a8 = a8
        self.scale = scale
        self.orig_shape = orig_shape
        self.dtype = dtype
        self.gs = gs

    @property
    def shape(self):
        return self.orig_shape

    def dequant(self):
        """Dequantize back to fp16/bf16 [*orig_shape] (boundary helper)."""
        M, G = self.scale.shape
        a8 = self.a8.reshape(M, G, self.gs).float()
        out = (a8 * self.scale.unsqueeze(-1)).reshape(self.a8.shape)
        out = out.to(self.dtype).reshape(self.orig_shape)
        if (
            not INT4XPULinear._nan_trace_logged
            and os.environ.get("OMNIXPU_INT4_NAN_TRACE", "0") != "0"
        ):
            if not torch.isfinite(out).all():
                INT4XPULinear._nan_trace_logged = True
                sc_f = self.scale.float()
                log.warning(
                    "[int4] NaN in W4ActS8.dequant: shape=%s dtype=%s "
                    "scale_max=%.4g scale_inf=%d scale_nan=%d a8_nonzero=%d",
                    tuple(out.shape), out.dtype,
                    sc_f.abs().max().item(),
                    int((~torch.isfinite(sc_f)).sum()),
                    int(torch.isnan(sc_f).sum()),
                    int(torch.count_nonzero(self.a8)),
                )
            elif torch.count_nonzero(out) == 0:
                INT4XPULinear._nan_trace_logged = True
                log.warning("[int4] ALL-ZERO W4ActS8.dequant: shape=%s dtype=%s", tuple(out.shape), out.dtype)
        return out

    # ── 张量协议兜底：任何边界运算（残差加法、标量乘、slice、函数调用等）
    #    遇到 W4ActS8 时自动反量化，保证通用闭包不依赖逐模型 patch。──────
    def __add__(self, other):
        return self.dequant().__add__(other)
    def __radd__(self, other):
        return self.dequant().__radd__(other)
    def __sub__(self, other):
        return self.dequant().__sub__(other)
    def __rsub__(self, other):
        return self.dequant().__rsub__(other)
    def __mul__(self, other):
        return self.dequant().__mul__(other)
    def __rmul__(self, other):
        return self.dequant().__rmul__(other)
    def __truediv__(self, other):
        return self.dequant().__truediv__(other)
    def __rtruediv__(self, other):
        return self.dequant().__rtruediv__(other)
    def __matmul__(self, other):
        return self.dequant().__matmul__(other)
    def __rmatmul__(self, other):
        return self.dequant().__rmatmul__(other)
    def __neg__(self):
        return self.dequant().__neg__()
    def __getitem__(self, key):
        return self.dequant().__getitem__(key)
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Let torch ops (F.silu, F.gelu, reshape, cat, ...) transparently
        dequantize. Falls back to the wrapped tensor when a given op is not
        in the safe list (avoids recursion on dequant itself)."""
        kwargs = kwargs or {}
        safe = {"dequant"}
        name = getattr(func, "__name__", "")
        if name in safe:
            return NotImplemented

        def _m(t):
            return t.dequant() if isinstance(t, W4ActS8) else t
        try:
            import torch
            new_args = tuple(_m(a) for a in args)
            new_kwargs = {k: _m(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
        except Exception:
            return NotImplemented

    def __getattr__(self, name):
        """Forward any other attribute/method (device, reshape, contiguous,
        to, float, ...) to the dequantized tensor. Only fires for names not in
        __slots__ (a8/scale/orig_shape/dtype/gs)."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.dequant(), name)


def _aimdo_manages():
    """AIMDO（XPU allocator）是否已接管显存管理。"""
    try:
        from .int4_xpu_aimdo import aimdo_active
        return aimdo_active()
    except Exception:
        return False


def _wrap_qwenimage_attn_shared_quant(model):
    """QwenImage w4a8: quantize the attention input once and feed the same
    W4ActS8 to all six input projections (to_q/k/v + add_q/k/v).

    Same math as per-layer quantization (identical input -> identical s8),
    but turns 6 quantize_act_s8 calls into 2 per block, and keeps the
    attention input resident as s8 instead of fp16.
    """
    try:
        from comfy.ldm.qwen_image.model import Attention as QwenAttn
    except Exception:
        return 0
    from omni_xpu_kernel import svdq

    n = 0
    for m in model.modules():
        if not isinstance(m, QwenAttn):
            continue
        to_q = getattr(m, "to_q", None)
        if not isinstance(to_q, INT4XPULinear) or getattr(to_q, "_backend", "w4a16") == "w4a16":
            continue
        gs = getattr(to_q, "_act_gs", 0) or (getattr(to_q, "_group_size", 64) or 64)
        orig_fwd = m.forward

        def _mk_fwd(orig, gs):
            def _to_s8(t):
                if isinstance(t, W4ActS8):
                    return t
                if not isinstance(t, torch.Tensor) or t.device.type != "xpu":
                    return t
                h = t.reshape(-1, t.shape[-1])
                if hasattr(svdq, "quantize_act_s8") and gs in (32, 64):
                    a8, sc = svdq.quantize_act_s8(h, gs)
                else:
                    a8, sc = _quantize_s8_act(h, gs)
                return W4ActS8(a8, sc, tuple(t.shape), t.dtype, gs)

            def fwd(hidden_states, encoder_hidden_states, *args, **kwargs):
                hidden_states = _to_s8(hidden_states)
                encoder_hidden_states = _to_s8(encoder_hidden_states)
                return orig(hidden_states, encoder_hidden_states, *args, **kwargs)

            return fwd

        m.forward = _mk_fwd(orig_fwd, gs)
        n += 1
    return n


def _wrap_qwenimage_gelu_s8(model):
    """QwenImage w4a8: keep the GELU input projection's output resident as s8
    (the 4x-width MLP intermediate is the single largest activation tensor),
    dequantizing only at the GELU boundary.
    """
    try:
        from comfy.ldm.qwen_image.model import GELU as QwenGELU
    except Exception:
        return 0

    n = 0
    for m in model.modules():
        if not isinstance(m, QwenGELU):
            continue
        proj = getattr(m, "proj", None)
        if not isinstance(proj, INT4XPULinear) or getattr(proj, "_backend", "w4a16") == "w4a16":
            continue
        object.__setattr__(proj, "_out_mode", "s8")

        def _mk_fwd(_proj, _approx):
            def fwd(hidden_states):
                hidden_states = _proj(hidden_states)
                if isinstance(hidden_states, W4ActS8):
                    hidden_states = hidden_states.dequant()
                return torch.nn.functional.gelu(
                    hidden_states, approximate=_approx
                )

            return fwd

        m.forward = _mk_fwd(m.proj, m.approximate)
        n += 1
    return n


def _wrap_qwenimage_block_s8(model):
    """QwenImage w4a8: keep the image-stream MLP output projection resident as
    s8 (out_mode="s8"), dequantizing only at the residual-gate boundary.

    The text-stream MLP output goes through addcmul and attn outputs through
    Dropout, which cannot consume W4ActS8 yet; those stay fp16.
    """
    n_lin = 0
    for m in model.modules():
        mlp = getattr(m, "img_mlp", None)
        if mlp is not None and len(getattr(mlp, "net", ())) >= 3:
            proj = mlp.net[2]
            if isinstance(proj, INT4XPULinear) and getattr(proj, "_backend", "w4a16") == "w4a8":
                object.__setattr__(proj, "_out_mode", "s8")
                n_lin += 1

    try:
        from comfy.ldm.qwen_image.model import QwenImageTransformerBlock
    except Exception:
        return n_lin

    n_gate = 0
    for m in model.modules():
        if not isinstance(m, QwenImageTransformerBlock):
            continue
        orig_gate = m._apply_gate

        def _mk_gate(orig):
            def fwd(x, y, gate, timestep_zero_index=None):
                if isinstance(x, W4ActS8):
                    x = x.dequant()
                return orig(x, y, gate, timestep_zero_index)

            return fwd

        m._apply_gate = _mk_gate(orig_gate)
        n_gate += 1
    return n_lin, n_gate


def _wrap_attn_probe(model):
    """QW attention 轻量探针：forward 前 reset_peak + 后 synchronize。

    旧独立插件用这个把 QW 显存约束在 13-14GB（AIMDO 下也有效）。机制：
    sync 强制 onednn/XPU 异步队列逐层完成，AIMDO 缓存块的 barrier 随之
    完成，ComfyUI 的 soft_empty_cache 才能真正把缓存池释放掉；reset_peak
    防止 torch 峰值统计被长序列持续推高。OMNIXPU_INT4_PROBE=0 关闭。
    """
    try:
        from comfy.ldm.qwen_image.model import Attention as QwenAttn
    except Exception:
        return 0
    n = 0
    for m in model.modules():
        if not isinstance(m, QwenAttn):
            continue
        orig_fwd = m.forward

        def _mk_fwd(orig):
            def fwd(*args, **kwargs):
                try:
                    torch.xpu.reset_peak_memory_stats()
                except Exception:
                    pass
                out = orig(*args, **kwargs)
                try:
                    torch.xpu.synchronize()
                except Exception:
                    pass
                return out

            return fwd

        m.forward = _mk_fwd(orig_fwd)
        n += 1
    return n


def _wrap_flux_s8(model):
    """FLUX-family (Krea2 / ZIT / Boogu when they use flux.layers): keep the
    output projections (attn.proj, mlp net[2], SingleStream linear2) resident
    as s8, dequantizing only at the apply_mod residual boundary.

    apply_mod is patched module-wide but only touches W4ActS8 inputs; all
    other callers are untouched.
    """
    try:
        import comfy.ldm.flux.layers as flux_layers
        from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
    except Exception:
        return 0

    _orig_apply_mod = flux_layers.apply_mod

    def _apply_mod_s8(tensor, m_mult, m_add=None, modulation_dims=None):
        if isinstance(tensor, W4ActS8):
            tensor = tensor.dequant()
        return _orig_apply_mod(tensor, m_mult, m_add, modulation_dims)

    flux_layers.apply_mod = _apply_mod_s8

    n = 0
    for m in model.modules():
        if isinstance(m, DoubleStreamBlock):
            for attr in ("img_attn", "txt_attn"):
                attn = getattr(m, attr, None)
                proj = getattr(attn, "proj", None)
                if isinstance(proj, INT4XPULinear) and getattr(proj, "_backend", "w4a16") == "w4a8":
                    object.__setattr__(proj, "_out_mode", "s8")
                    n += 1
            for mlp_attr in ("img_mlp", "txt_mlp"):
                mlp = getattr(m, mlp_attr, None)
                if mlp is not None and len(mlp) >= 3:
                    p = mlp[2]
                    if isinstance(p, INT4XPULinear) and getattr(p, "_backend", "w4a16") == "w4a8":
                        object.__setattr__(p, "_out_mode", "s8")
                        n += 1
        elif isinstance(m, SingleStreamBlock):
            lin2 = getattr(m, "linear2", None)
            if isinstance(lin2, INT4XPULinear) and getattr(lin2, "_backend", "w4a16") == "w4a8":
                object.__setattr__(lin2, "_out_mode", "s8")
                n += 1
    return n


_OUTPUT_PROJ_ROLES = {
    "wo", "out", "out_proj", "o_proj", "proj_out", "to_out", "to_out.0",
    "down", "fc2", "w2", "linear2", "linear_out", "proj",
}
_INPUT_PROJ_ROLES = {
    "wq", "wk", "wv", "gate", "up", "w1", "w3", "fc1",
    "to_q", "to_k", "to_v", "to_gate", "qkv", "qkv_proj",
}


def _auto_s8_closure(model):
    """Generic layer-to-layer s8 closure for any architecture (w4a8 backend).

    Design note (measured on Krea2): output projections (wo/out/down) are NOT
    set to s8. Their result feeds the residual stream which is fp16 by model
    semantics, so s8-ing them only adds a quantize+dequant round-trip per
    layer. The memory win is on the *input* side:
      1. Multi-input attention (wq/wk/wv [+gate]) is wrapped to quantize the
         shared input once; every input projection consumes the same s8 copy
         (Krea2: 8 per-layer quantizations -> 3).
      2. SwiGLU (gate/up/down) is wrapped so silu(gate)*up is quantized to s8
         before the down projection; the 4x-width intermediate (the largest
         activation) stays s8 across the down GEMM instead of fp16.
      3. W4ActS8's tensor protocol remains as a safety net for any residual
         boundary that receives an s8 tensor (dequantizes transparently).

    Returns dict {out_s8, attn_shared, swiglu} of patched counts.
    """
    import os as _os
    from omni_xpu_kernel import svdq
    from omni_xpu_kernel import int8 as int8_ops

    def _w4(x, gs):
        h = x.reshape(-1, x.shape[-1])
        if h.dtype not in (torch.float16, torch.bfloat16):
            h = h.to(torch.float16)
        if h.shape[-1] % gs != 0:
            import traceback
            frames = traceback.extract_stack()[-8:]
            chain = " <- ".join(
                f"{f.filename.split(chr(92))[-1]}:{f.lineno}:{f.name}"
                for f in frames)
            log.warning("[int4] _w4: shape=%s K=%d gs=%d mismatch -> fp16 passthrough "
                        "| %s", tuple(x.shape), h.shape[-1], gs, chain)
            return x
        if hasattr(svdq, "quantize_act_s8") and gs in (32, 64):
            a8, sc = svdq.quantize_act_s8(h, gs)
        else:
            a8, sc = _quantize_s8_act(h, gs)
        return W4ActS8(a8, sc, tuple(x.shape), x.dtype, gs)

    _sw_count = [0]
    counts = {"out_s8": 0, "attn_shared": 0, "swiglu": 0}
    lin_by_id = {}
    for m in model.modules():
        if isinstance(m, INT4XPULinear):
            lin_by_id[id(m)] = m

    # ── 1. 多输入投影 attention：共享量化一次 ─────────────────────────────
    for m in list(model.modules()):
        projs = []
        for attr in ("wq", "wk", "wv", "gate", "to_q", "to_k", "to_v"):
            p = getattr(m, attr, None)
            if isinstance(p, INT4XPULinear) and getattr(p, "_backend", "w4a16") == "w4a8":
                projs.append((attr, p))
        if len(projs) < 3:
            continue
        gs = next((getattr(p, "_act_gs", 0) or (getattr(p, "_group_size", 64) or 64)
                   for _, p in projs), 64)
        orig_fwd = m.forward

        def _mk_attn_fwd(orig, _gs):
            def fwd(x, *args, **kwargs):
                if not isinstance(x, W4ActS8):
                    x = _w4(x, _gs)
                return orig(x, *args, **kwargs)
            return fwd

        m.forward = _mk_attn_fwd(orig_fwd, gs)
        counts["attn_shared"] += 1

    # ── 2. SwiGLU（gate/up/down 或 w1/w3/w2 / fc1/fc2/fc3）融合 ───────────
    for m in list(model.modules()):
        triples = []
        if all(hasattr(m, a) for a in ("gate", "up", "down")):
            triples.append(("gate", "up", "down"))
        if all(hasattr(m, a) for a in ("w1", "w3", "w2")):
            triples.append(("w1", "w3", "w2"))
        if all(hasattr(m, a) for a in ("fc1", "fc2", "fc3")):
            triples.append(("fc1", "fc2", "fc3"))
        for ga, ua, da in triples:
            g, u, d = getattr(m, ga), getattr(m, ua), getattr(m, da)
            if not all(isinstance(p, INT4XPULinear) and getattr(p, "_backend", "w4a16") == "w4a8"
                       for p in (g, u, d)):
                continue
            gs = getattr(g, "_act_gs", 0) or (getattr(g, "_group_size", 64) or 64)
            orig_fwd = m.forward

            def _mk_swiglu_fwd(orig, _g, _u, _d, _gs, _gidx):
                def fwd(x):
                    x = _w4(x, _gs) if not isinstance(x, W4ActS8) else x
                    g = _g(x)                       # fp16 [M, 4D] transient
                    u = _u(x)                       # fp16 [M, 4D] transient
                    # 融合 SiLU(g)*u -> s8 + group scale：4x 中间不物化 fp16。
                    # 走 group-wise kernel（scale [M, K/gs] 匹配 onednn_s8u4）。
                    # groupwise kernel：silu(g)*u 直接量化，4x fp16 中间不物化。
                    # 默认关闭（QW 不走此路径；Krea2 类 SwiGLU 需要时显式开
                    # OMNIXPU_INT4_GROUPWISE=1）。
                    if (hasattr(int8_ops, "fused_silu_mul_quantize_groupwise")
                            and _os.environ.get("OMNIXPU_INT4_GROUPWISE", "0") != "0"):
                        a8, sc = int8_ops.fused_silu_mul_quantize_groupwise(
                            g.reshape(-1, g.shape[-1]),
                            u.reshape(-1, u.shape[-1]), _gs)
                        del g, u
                        out_shape = tuple(x.orig_shape[:-1]) + (a8.shape[-1],)
                        return _d(W4ActS8(a8, sc, out_shape, x.dtype, _gs))
                    h = torch.nn.functional.silu(g).mul_(u)   # fp16 4x transient
                    del g, u
                    h_shape = tuple(x.orig_shape[:-1]) + (h.shape[-1],)
                    a8, sc = _q_parts(h, _gs)
                    return _d(W4ActS8(a8, sc, h_shape, x.dtype, _gs))
                return fwd

            m.forward = _mk_swiglu_fwd(orig_fwd, g, u, d, gs, _sw_count[0])
            _sw_count[0] += 1
            counts["swiglu"] += 1
            break
    return counts


def _q_parts(x, gs):
    """Return (a8, scale) for a fp16 tensor (group-wise, matches onednn_s8u4)."""
    from omni_xpu_kernel import svdq
    h = x.reshape(-1, x.shape[-1])
    if h.dtype not in (torch.float16, torch.bfloat16):
        h = h.to(torch.float16)
    if hasattr(svdq, "quantize_act_s8") and gs in (32, 64):
        return svdq.quantize_act_s8(h, gs)
    return _quantize_s8_act(h, gs)


def _quantize_s8_act(x, group_size):
    """Per-group symmetric s8 activation quantize.

    Returns (act_s8 [M,K] int8, xscales [M,G] f16) for onednn_s8u4_gemm.
    """
    M, K = x.shape
    G = K // group_size
    xf = x.float()
    gmax = xf.view(M, G, group_size).abs().amax(dim=2)  # [M, G]
    scale = (gmax / 127.0).clamp_min(1e-10)
    a8 = (xf.view(M, G, group_size) / scale.unsqueeze(-1)).round().clamp(-127, 127)
    # scale f32（f16 会对 QW 大激活溢出为 inf -> dequant NaN -> 黑图）
    return a8.to(torch.int8).view(M, K), scale.float().contiguous()


class INT4XPULinear(nn.Module):
    _call_count = 0
    _nan_trace_logged = False
    # 预热：首次 forward（采样开始，CLIP 已卸载）一次性搬入所有权重，
    # 消除逐层 .to(dev) 造成的显存爬升（QW 839 层 / 9.5GB 会从 ~7GB
    # 一路爬到 ~10.5GB，这就是"显存跳跃"的直接来源）。
    _prewarm_target = None
    _prewarm_done = False

    def __init__(self, in_features, out_features, w_int4, w_scales, bias=None,
                 use_quarot=False, hadamard_H=None, group_size=128,
                 act_dtype=torch.float16, backend="w4a16", quarot_gs=None,
                 out_mode="fp16", correction=None, zp=None, tint4_mode=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        object.__setattr__(self, "w_int4", w_int4)
        object.__setattr__(self, "w_scales", w_scales)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._prepared = False
        object.__setattr__(self, "_wa4_lora_entries", None)
        object.__setattr__(self, "_use_quarot", use_quarot)
        object.__setattr__(self, "_hadamard_H", hadamard_H)
        # 权重量化分组（每层 w4a4_group_size，32/64）与 QuaRot 旋转分组
        # （__w4a4_quarot_group_size__，通常 128）是两个概念，分开存储。
        object.__setattr__(self, "_group_size", group_size)
        # 激活量化分组：a8 下 src 侧 oneDNN 只稳定支持 32/64；
        # 权重 gs>64（如 tint4 gs=128）时激活按 64 量化，
        # kernel 的 onednn_s8u4_gemm 已支持 src/wei 分组解耦。
        _g = group_size or 64
        object.__setattr__(
            self, "_act_gs",
            64 if (backend in ("w4a8", "w4a8-s8", "w4a8-88") and _g > 64) else _g,
        )
        object.__setattr__(self, "_quarot_gs", quarot_gs or group_size)
        object.__setattr__(self, "_act_dtype", act_dtype)
        object.__setattr__(self, "_backend", backend)
        # TINT4 非对称 zp 修正项 [N] f32（None = 对称 wa4 模型）
        object.__setattr__(self, "_correction", correction)
        # TINT4 原生模式：raw u4 字节视图 + per-block zp，直接喂
        # onednn_int4_gemm_tint4（无转换、无修正项）
        object.__setattr__(self, "_zp", zp)
        object.__setattr__(self, "_tint4_mode", bool(tint4_mode))
        # LoRA GPU 缓存：(tensor_map, id(entries), (dev, dtype))——LoRA 权重
        # 只搬一次 GPU，避免每层每次 forward 都 H2D（QW 挂 LoRA 时每步
        # 数百次 H2D + CPU staging，实测 CPU 打满）
        object.__setattr__(self, "_wa4_lora_gpu", None)
        # "fp16" (default) keeps the classic behaviour; "s8" returns W4ActS8
        # from w4a8 forward so chained INT4XPULinear layers can skip re-quantizing.
        object.__setattr__(self, "_out_mode", out_mode)

    def _preload_to_xpu(self, dev):
        """Prepare + move this layer's weights to dev (used by prewarm)."""
        self._prepare()
        if self._w_packed.device != dev:
            self._w_packed = self._w_packed.to(dev)
            self._w_s = self._w_s.to(dev)
            if getattr(self, "_zp", None) is not None and self._zp.device != dev:
                self._zp = self._zp.to(dev)
            if getattr(self, "_w_s8", None) is not None and self._w_s8.device != dev:
                self._w_s8 = self._w_s8.to(dev)

    @staticmethod
    def _prewarm_once(dev):
        """First-call bulk weight preload; no-op afterwards."""
        if INT4XPULinear._prewarm_done:
            return
        INT4XPULinear._prewarm_done = True
        target = INT4XPULinear._prewarm_target
        if target is None:
            log.info("[int4] prewarm: no target (skip)")
            return
        _t0 = time.perf_counter()
        _n = 0
        for m in target.modules():
            if isinstance(m, INT4XPULinear):
                try:
                    m._preload_to_xpu(dev)
                    _n += 1
                except Exception:
                    pass
        log.info("[int4] prewarm: %d layers moved to %s in %.2fs",
                 _n, dev, time.perf_counter() - _t0)

    def _prepare(self):
        if self._prepared:
            return
        from omni_xpu_kernel import svdq
        backend = getattr(self, "_backend", "w4a16")
        if getattr(self, "_tint4_mode", False) or self._correction is not None:
            # TINT4 原生：int32 qdata 的字节视图就是 packed u4（小端 nibble
            # 顺序与 oneDNN u4 布局一致），零转换零拷贝。
            # tint4-a8（转换路径）：同样用 raw 视图（不 XOR），oneDNN 标量
            # zp=8 得 q-8，(8-zp)*scale 修正项由 forward 的 correction 分支补。
            self._w_packed = self.w_int4.view(torch.uint8).contiguous()
            self._w_s = self.w_scales
            # 释放 CPU 侧原始引用：_w_packed 是字节视图（持同一 storage），
            # 一旦 prewarm 把 _w_packed 搬到 GPU，CPU 10GB int32 即被释放——
            # 否则 tint4 会 GPU 10GB + CPU 10GB 并存（RAM +10GB 的来源）。
            object.__setattr__(self, "w_int4", None)
            object.__setattr__(self, "w_scales", None)
        else:
            w_u8 = self.w_int4.to(torch.uint8)
            self._w_packed, self._w_s = svdq.prepare_onednn_weights(w_u8, self.w_scales)
            # 权重只保留打包态：原始 u4 与 scales 的 CPU 副本立即释放。
            # （WINT4/tint4 参考实现：权重始终以打包态存储，forward 时再解包，
            #  避免加载期 CPU 物化多份副本造成 RAM 峰值。）
            object.__setattr__(self, "w_int4", None)
            object.__setattr__(self, "w_scales", None)
            del w_u8
        if backend in ("w4a8-s8", "w4a8-88"):
            # 88: unpack centered u4 (u4-8) to exact s8 weights [N, K] once.
            # unpack_int4 is an ESIMD kernel that reads device memory; the
            # packed tensor must be on XPU first (host pointers crash L0).
            w_packed_dev = self._w_packed.to("xpu")
            self._w_s8 = (svdq.unpack_int4(w_packed_dev, signed=False)
                          .to(torch.int8) - 8).contiguous()
            del w_packed_dev
        self._prepared = True

    def release_xpu(self):
        try:
            if self._w_packed is not None and self._w_packed.device.type == 'xpu':
                self._w_packed = self._w_packed.to('cpu')
            if self._w_s is not None and self._w_s.device.type == 'xpu':
                self._w_s = self._w_s.to('cpu')
            if getattr(self, "_zp", None) is not None and self._zp.device.type == 'xpu':
                self._zp = self._zp.to('cpu')
            if getattr(self, '_w_s8', None) is not None and self._w_s8.device.type == 'xpu':
                self._w_s8 = self._w_s8.to('cpu')
            object.__setattr__(self, "_wa4_lora_gpu", None)
        except Exception:
            pass

    def ensure_xpu(self, dev="xpu"):
        """Prepare + move packed weights onto the device immediately.

        Called once after model load so the CPU does not keep a ~GB-scale
        packed-weight copy resident while the CLIP/VAE loads next (that is
        the main reason Qwen runs push system RAM to 40GB+). Weights stay
        wa4-managed (AIMDO does not touch them either way).
        """
        self._prepare()
        try:
            if self._w_packed.device.type != dev:
                self._w_packed = self._w_packed.to(dev)
                self._w_s = self._w_s.to(dev)
            if getattr(self, '_w_s8', None) is not None and self._w_s8.device.type != dev:
                self._w_s8 = self._w_s8.to(dev)
        except Exception:
            pass

    def forward(self, x):
        INT4XPULinear._prewarm_once(getattr(x, "device", None))
        self._prepare()
        from omni_xpu_kernel import svdq
        out_mode = getattr(self, "_out_mode", "fp16")
        s8_in = isinstance(x, W4ActS8)
        if s8_in:
            dev = x.a8.device
            s = x.orig_shape
            x_dtype = x.dtype
        else:
            dev = x.device
            s = x.shape
            x_dtype = x.dtype
        if self._w_packed.device != dev:
            self._w_packed = self._w_packed.to(dev)
            self._w_s = self._w_s.to(dev)
            if getattr(self, "_zp", None) is not None and self._zp.device != dev:
                self._zp = self._zp.to(dev)
            if getattr(self, '_w_s8', None) is not None and self._w_s8.device != dev:
                self._w_s8 = self._w_s8.to(dev)
        # ── 后端选择：w4a16 / w4a8(84) / w4a8-s8(88) / w4a4(禁用) ─────
        # w4a16: fp16 激活直接进 oneDNN u4 GEMM（免量化，精度最好）。
        # w4a8(84) : 激活按组量化到 s8 -> oneDNN s8u4 GEMM（权重保持 u4）。
        # w4a8-s8(88): 激活 s8 + 权重解包成 s8 -> oneDNN s8x8 GEMM（更快引擎，
        #              但权重内存 x2；u4 持久、s8 按层缓存可折中）。
        # w4a4 : 禁用（层间 INT4 传递未实现）。
        # out_mode="s8"（仅 w4a8）：返回 W4ActS8，层间激活保持 s8。
        backend = getattr(self, "_backend", "w4a16")
        gs = getattr(self, "_act_gs", 0) or (getattr(self, "_group_size", 64) or 64)
        if backend == "w4a4":
            # 层间 4-bit 传递尚未实现：当前会物化 s8 中间张量，a4 无内存/带宽收益
            # 且精度最差。暂屏蔽，待层间 u4 传递功能完成后重新开放。
            raise NotImplementedError(
                "[int4] w4a4 backend is disabled: layer-to-layer INT4 activation "
                "transfer is not implemented yet. Use w4a8 or w4a16."
            )
        if backend == "w4a16":
            if s8_in:
                x2 = x.dequant().reshape(-1, s[-1])
            else:
                x2 = x.reshape(-1, s[-1]).to(self._act_dtype)
            if self._use_quarot and self._hadamard_H is not None:
                try:
                    from .int4_xpu_quarot import rotate_activation
                    x2 = rotate_activation(
                        x2, self._hadamard_H,
                        getattr(self, "_quarot_gs", self._group_size))
                except Exception:
                    pass
            if getattr(self, "_tint4_mode", False):
                # TINT4 原生：per-block zp 在 oneDNN 内应用，w=(q-zp)*scale
                # A 系列 fork：独立 onednn_int4_gemm_tint4 / torchao 名；
                # B 系列（原仓库 PR）：onednn_int4_gemm_preconverted 的 zp 参数。
                if hasattr(svdq, "onednn_int4_gemm_tint4"):
                    o = svdq.onednn_int4_gemm_tint4(
                        x2, self._w_packed, self._zp, self._w_s)
                elif hasattr(svdq, "onednn_int4_gemm_torchao"):
                    o = svdq.onednn_int4_gemm_torchao(
                        x2, self._w_packed, self._zp, self._w_s)
                else:
                    o = svdq.onednn_int4_gemm_preconverted(
                        x2, self._w_packed, self._w_s, self._zp)
            else:
                o = svdq.onednn_int4_gemm_preconverted(x2, self._w_packed, self._w_s)
        else:
            if s8_in:
                a8, xsc = x.a8, x.scale
                x2 = None  # 惰性：LoRA 需要 fp16 激活时才反量化
            else:
                x2 = x.reshape(-1, s[-1]).to(self._act_dtype)
                if self._use_quarot and self._hadamard_H is not None:
                    try:
                        from .int4_xpu_quarot import rotate_activation
                        x2 = rotate_activation(
                            x2, self._hadamard_H,
                            getattr(self, "_quarot_gs", self._group_size))
                    except Exception:
                        pass
                if hasattr(svdq, "quantize_act_s8") and gs in (32, 64):
                    a8, xsc = svdq.quantize_act_s8(x2, gs)
                else:
                    a8, xsc = _quantize_s8_act(x2, gs)
            if backend in ("w4a8-s8", "w4a8-88"):
                o = svdq.onednn_s8s8_gemm(a8, xsc, self._w_s8, self._w_s,
                                          self._act_dtype)
            else:
                o = svdq.onednn_s8u4_gemm(a8, xsc,
                                          self._w_packed, self._w_s,
                                          self._act_dtype)

        # ── TINT4 非对称 zp 修正：w=(q-zp)*scale = q'*scale + (8-zp)*scale ──
        if (self._correction is not None
                and not getattr(self, "_tint4_mode", False)
                and os.environ.get("OMNIXPU_INT4_T4A8_NOCORR", "0") == "0"):
            try:
                if backend == "w4a16":
                    act_2d = x2.reshape(-1, s[-1])
                else:
                    if s8_in:
                        act_raw = x.dequant().reshape(-1, s[-1])
                    else:
                        act_raw = x2.reshape(-1, s[-1])
                    act_2d = act_raw
                # 按组求和 [M, G]，再乘 per-group 修正系数 [G, N]（小矩阵乘）
                Gc = self._correction.shape[0]
                gs_c = act_2d.shape[1] // Gc
                act_gsum = act_2d.float().view(-1, Gc, gs_c).sum(dim=2)
                corr = self._correction.to(dev, dtype=torch.float32)
                o = o + (act_gsum @ corr).to(o.dtype)
            except Exception:
                pass

        entries = self._wa4_lora_entries
        if entries is not None and len(entries) > 0:
            cd = self._act_dtype
            if x2 is None:
                x2 = x.dequant().reshape(-1, s[-1])
            # LoRA GPU 缓存：entries 不变 + 设备/精度不变时复用已搬好的
            # 张量（只搬一次）；entries 被替换（新 LoRA 栈）或换设备时重建。
            _gcache = self._wa4_lora_gpu
            if _gcache is None or _gcache[1] != id(entries) or _gcache[2] != (dev, cd):
                _gcache = ({}, id(entries), (dev, cd))
                object.__setattr__(self, "_wa4_lora_gpu", _gcache)
            _gmap = _gcache[0]
            for lora_entries in entries.values():
                for entry in lora_entries:
                    etype = entry[0] if isinstance(entry[0], str) else None
                    if etype == "delta":
                        _, delta_cpu, mult = entry[:3]
                        d = _gmap.get(id(delta_cpu))
                        if d is None:
                            d = delta_cpu.to(dev, dtype=cd)
                            _gmap[id(delta_cpu)] = d
                        lo = x2 @ d.T * mult
                        if lo.shape[1] == o.shape[1]:
                            o = o + lo.to(o.dtype)
                        del lo, d
                    else:
                        A_cpu, B_cpu, mult = entry[:3]
                        Ad = _gmap.get(("A", id(A_cpu)))
                        if Ad is None:
                            Ad = A_cpu.to(dev, dtype=cd)
                            _gmap[("A", id(A_cpu))] = Ad
                        Bd = _gmap.get(("B", id(B_cpu)))
                        if Bd is None:
                            Bd = B_cpu.to(dev, dtype=cd)
                            _gmap[("B", id(B_cpu))] = Bd
                        lo = (x2 @ Ad.T) @ Bd.T * mult
                        if lo.shape[1] == o.shape[1]:
                            o = o + lo.to(o.dtype)
                        del lo, Ad, Bd
            del x2

        if self.bias is not None:
            if self.bias.device != dev or self.bias.dtype != x_dtype:
                self.bias.data = self.bias.data.to(device=dev, dtype=x_dtype)
            o = o + self.bias

        if out_mode == "s8" and backend != "w4a16":
            if hasattr(svdq, "quantize_act_s8") and gs in (32, 64):
                o_s8, o_sc = svdq.quantize_act_s8(o.reshape(-1, o.shape[-1]), gs)
            else:
                o_s8, o_sc = _quantize_s8_act(o.reshape(-1, o.shape[-1]), gs)
            out_shape = s[:-1] + (o.shape[-1],)
            INT4XPULinear._call_count += 1
            if INT4XPULinear._call_count % 500 == 0:
                log.info("[int4] %d forward calls, xpu mem: %dMB (s8-out)",
                         INT4XPULinear._call_count,
                         torch.xpu.memory_allocated() // 1024 // 1024)
            if _empty_cache_enabled():
                _empty_cache()
            _sync_point()
            if (
                not INT4XPULinear._nan_trace_logged
                and os.environ.get("OMNIXPU_INT4_NAN_TRACE", "0") != "0"
                and not torch.isfinite(o).all()
            ):
                INT4XPULinear._nan_trace_logged = True
                log.warning(
                    "[int4] NaN in s8-out GEMM output (call=%d backend=%s "
                    "out=%s in=%s s8_in=%s)",
                    INT4XPULinear._call_count, backend, tuple(o.shape),
                    tuple(s), s8_in,
                )
            if (
                not INT4XPULinear._nan_trace_logged
                and os.environ.get("OMNIXPU_INT4_NAN_TRACE", "0") != "0"
                and torch.isfinite(o).all()
                and o.numel() > 0
                and torch.count_nonzero(o) == 0
            ):
                INT4XPULinear._nan_trace_logged = True
                log.warning(
                    "[int4] ALL-ZERO s8-out GEMM output (call=%d backend=%s "
                    "out=%s in=%s s8_in=%s)",
                    INT4XPULinear._call_count, backend, tuple(o.shape),
                    tuple(s), s8_in,
                )
            return W4ActS8(o_s8, o_sc, out_shape, x_dtype, gs)

        if len(s) == 3:
            o = o.reshape(*s[:-1], -1)
        o = o.to(x_dtype)
        INT4XPULinear._call_count += 1
        if INT4XPULinear._call_count % 500 == 0:
            log.info("[int4] %d forward calls, xpu mem: %dMB",
                     INT4XPULinear._call_count, torch.xpu.memory_allocated() // 1024 // 1024)
        if _empty_cache_enabled():
            _empty_cache()
        _sync_point()
        if (
            not INT4XPULinear._nan_trace_logged
            and os.environ.get("OMNIXPU_INT4_NAN_TRACE", "0") != "0"
            and isinstance(o, torch.Tensor)
            and not torch.isfinite(o).all()
        ):
            INT4XPULinear._nan_trace_logged = True
            log.warning(
                "[int4] NaN detected in fp16-out path (call=%d backend=%s "
                "out=%s in=%s s8_in=%s)",
                INT4XPULinear._call_count, backend, tuple(o.shape),
                tuple(s), s8_in,
            )
        return o


def _normalize_index_path(name: str) -> str | None:
    for pf in ("diffusion_model.", "model.diffusion_model.", "model."):
        if name.startswith(pf): name = name[len(pf):]; break
    if name.startswith("img_in") or name.startswith("final_layer"): return None
    for old, new in [("layers.", "blocks."), ("joint_blocks.", "blocks."),
                     ("transformer_blocks.", "blocks."), ("double_blocks.", "blocks."),
                     ("single_blocks.", "blocks.")]:
        if name.startswith(old): name = new + name[len(old):]; break
    for a, b in [
        (".ff.", ".mlp."), (".feed_forward.", ".mlp."), (".ffn.", ".mlp."),
        (".attention.", ".attn."),
        (".to_q", ".wq"), (".to_k", ".wk"), (".to_v", ".wv"),
        (".to_out.0", ".wo"), (".to_out", ".wo"),
        (".out_proj", ".wo"), (".attn.out", ".attn.wo"),
        (".attn.proj", ".attn.wo"), (".attn.o_proj", ".attn.wo"),
        (".self_attn.q", ".attn.wq"), (".self_attn.k", ".attn.wk"),
        (".self_attn.v", ".attn.wv"), (".self_attn.o", ".attn.wo"),
        (".to_gate", ".gate"),
        (".q_proj", ".wq"), (".k_proj", ".wk"), (".v_proj", ".wv"),
        (".gate_proj", ".gate"), (".up_proj", ".w1"), (".down_proj", ".w2"),
        (".fc1", ".w1"), (".fc2", ".w2"), (".fc3", ".w3"),
    ]:
        name = name.replace(a, b)
    return name


def _build_wa4_lora_index(dm: nn.Module) -> dict:
    index: dict[str, nn.Module] = {}
    for name, module in dm.named_modules():
        if not isinstance(module, (INT4XPULinear, nn.Linear)):
            continue
        norm = _normalize_index_path(name)
        if norm is None:
            continue
        if norm.endswith(".attn.qkv") and isinstance(module, INT4XPULinear):
            out_f = module.out_features
            if out_f % 3 == 0:
                hs = out_f // 3
                base = norm.rsplit(".attn.qkv", 1)[0]
                index[f"{base}.attn.wq"] = module
                index[f"{base}.attn.wk"] = module
                index[f"{base}.attn.wv"] = module
        elif norm.endswith(".attn.qkv"):
            w = getattr(module, 'weight', None)
            out_f = w.shape[0] if w is not None and hasattr(w, 'shape') else 0
            if out_f > 0 and out_f % 3 == 0:
                hs = out_f // 3
                base = norm.rsplit(".attn.qkv", 1)[0]
                index[f"{base}.attn.wq"] = module
                index[f"{base}.attn.wk"] = module
                index[f"{base}.attn.wv"] = module
        index[norm] = module
    return index


def _convert_tint4_to_wa4(qdata, scale, zp, gs, signed_xor=True):
    """Convert TINT4/torchao Int4PlainInt32Tensor to the wa4 packed format.

    torchao semantics: w = (q - zp) * scale, q in [0,15] unsigned, zp per block.
    wa4/oneDNN stores signed int4 (center) with scalar zp=8, so we rewrite:
      q' = q - 8  (signed int4 [-8,7], stored as wa4 low-nibble packed)
      w = (q - zp) * scale = q' * scale + (8 - zp) * scale
    The (8-zp)*scale term is a per-output-column constant; the GEMM output is
    corrected by subtracting  outer(act_sum, correction) at forward time.

    Returns (packed_u4 [N, K/2] uint8, scale_wa4 [G, N] f16,
             correction [G, N] f32) where correction[g,n] = (8-zp)*scale.

    打包捷径：每个 int32 的 4 字节按小端恰好是 4 个 wa4 packed byte
    （低/高 nibble 正好对应 wa4 的 even/odd 配对）。a16 需要 signed 形态
    （q XOR 8，oneDNN 标量 zp=8 还原）；a8 走 raw 形态（不 XOR）：
    oneDNN zp=8 直接得 q-8，再加 (8-zp)*scale 修正项还原 tint4 语义。

    返回的 tensor 仍在设备上（GPU），由 _index_int4_from_sd 批量同步后统一
    搬回 CPU——避免 839 层 × 3 次 .cpu() 的逐层设备同步（实测 43.6s）。
    """
    dev = "xpu" if torch.xpu.is_available() else "cpu"
    qdata = qdata.contiguous()
    out_f = qdata.shape[0]
    in_f = qdata.shape[1] * 8
    blocks = in_f // gs
    if signed_xor:
        if dev == "xpu":
            packed = qdata.view(torch.uint8).to(dev, non_blocking=True)
            packed.bitwise_xor_(0x88)  # 原地翻转符号位，免额外 10GB 分配
        else:
            packed = (qdata.view(torch.uint8) ^ 0x88)
    else:
        packed = qdata.view(torch.uint8).to("cpu") if dev == "xpu" else qdata.view(torch.uint8)
    # scale/corr 小数据，留在 CPU 算（float 多线程，免 GPU 往返）
    scale_wa4 = scale.to(torch.float16).reshape(blocks, out_f).contiguous()
    correction = ((8.0 - zp.float()) * scale.float()).reshape(blocks, out_f).contiguous()
    return packed, scale_wa4, correction


def _index_int4_from_sd(sd, backend="w4a16", mode="kernel"):
    quant = {}
    _t0 = _tphase("index_int4 enter")
    _n_conv = 0
    _t_conv = 0.0
    _gpu_conv = []  # (norm, base, gs, packed_gpu, scale_gpu, corr_gpu)
    _FLUSH_EVERY = 64

    def _flush_gpu_conv():
        """把已排队转换的 GPU packed 非阻塞搬回 CPU（D2H 流水线化）。"""
        if not _gpu_conv:
            return
        _tc1 = time.time()
        for norm, base, gs_t, packed, scale_wa4, corr in _gpu_conv:
            quant[norm] = (
                packed.to("cpu", non_blocking=True),
                scale_wa4, packed.shape[0],
                packed.shape[1] * 2, base, gs_t, corr, None,
            )
        n = len(_gpu_conv)
        _gpu_conv.clear()
        if _TIMING:
            log.info("[wa4-timing] tint4 d2h flush: %.2fs (%d layers)",
                     time.time() - _tc1, n)

    for key in list(sd.keys()):
        if not key.endswith(".weight"): continue
        w = sd[key]
        if w.ndim != 2: continue
        base = key[:-7]
        # ── TINT4/torchao：int32 qdata + per-block zp → 转 wa4 ──
        # 仅 int32 qdata（量化层）走转换；BF16 未量化层走正常路径。
        is_tint4 = w.dtype == torch.int32 or sd.get(f"{base}.weight_zp") is not None
        if is_tint4:
            zp_t = sd.get(f"{base}.weight_zp")
            sc_t = sd.get(f"{base}.weight_scale")
            b1_t = sd.get(f"{base}.weight_b1")
            if zp_t is not None and sc_t is not None:
                try:
                    gs_t = int(b1_t.item()) if b1_t is not None else 128
                    norm = _normalize_index_path(base)
                    if norm is None:
                        continue
                    if (
                        _TINT4_NATIVE and backend == "w4a16"
                        and mode in ("kernel", "torchao")
                    ):
                        # 原生模式（a16）：raw int32 qdata（_prepare 里 view 成
                        # u4）+ per-block zp + fp16 scale，零转换、零 GPU 往返。
                        # torchao 回退也用同样的 raw 数据。
                        scale_wa4 = sc_t.to(torch.float16).contiguous()
                        zp_u8 = zp_t.to(torch.uint8).contiguous()
                        quant[norm] = (
                            w, scale_wa4, w.shape[0], w.shape[1] * 8,
                            base, gs_t, None, zp_u8,
                        )
                        for _aux in ("weight_zp", "weight_scale", "weight_b0",
                                     "weight_b1", "comfy_quant"):
                            sd.pop(f"{base}.{_aux}", None)
                        continue
                    _tc0 = time.time()
                    packed, scale_wa4, corr = _convert_tint4_to_wa4(
                        w, sc_t, zp_t, gs_t, signed_xor=(backend != "w4a8")
                    )
                    _n_conv += 1
                    _t_conv += time.time() - _tc0
                except Exception as _e:
                    log.warning("[int4] TINT4 convert failed %s: %r", base, _e)
                    continue
                _gpu_conv.append((norm, base, gs_t, packed, scale_wa4, corr))
                # 转换后立即消费 tint4 辅助键，避免它们进入模型构建触发
                # 超长的 "unexpected" 警告（839 层 × 5 个键）
                for _aux in ("weight_zp", "weight_scale", "weight_b0",
                             "weight_b1", "comfy_quant"):
                    sd.pop(f"{base}.{_aux}", None)
                if len(_gpu_conv) >= _FLUSH_EVERY:
                    _flush_gpu_conv()
                continue
        # TINT4/torchao 格式检测：qdata 是 int32，带 weight_zp/comfy_quant。
        # wa4 GEMM 期望 uint8 packed（每字节 2 个 int4）+ 标量 zp=8，
        # 直接索引会把 int32 qdata 截断成 uint8，产生错误结果，必须跳过。
        if (
            w.dtype == torch.int32
            or sd.get(f"{base}.weight_zp") is not None
            or sd.get(f"{base}.comfy_quant") is not None
        ):
            log.warning(
                "[int4] Skipping %s: TINT4/torchao format is not compatible "
                "with the wa4 loader (int32 qdata + per-block zero point)",
                base,
            )
            continue
        if w.dtype not in (torch.int8, torch.uint8):
            continue
        w_scale = sd.get(f"{base}.weight_scale")
        if w_scale is None or w_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32): continue
        gs = sd.get(f"{base}.w4a4_group_size", torch.tensor([64], dtype=torch.int32)).item()
        norm = _normalize_index_path(base)
        if norm is None: continue
        quant[norm] = (w, w_scale, w.shape[0], w_scale.shape[0] * gs, base, gs, None, None)
    # ── TINT4 转换结果批量搬回 CPU：一次同步，避免 839 层逐层 .cpu() ──
    _flush_gpu_conv()
    try:
        torch.xpu.synchronize()
    except Exception:
        pass
    _tphase("index_int4 done", _t0)
    if _TIMING and _n_conv:
        log.info("[wa4-timing] tint4 conversions: %d layers, wall %.2fs (avg %.3fs)",
                 _n_conv, _t_conv, _t_conv / _n_conv)
    log.info("[int4] Indexed %d INT4 groups", len(quant))
    return quant


def _patch_omni_norm(model, cfg_type):
    """omni norm V3：Krea2 scale+1.0 + wrapper 跳过"""
    if cfg_type in _OMNI_NORM_SKIP:
        log.info("[int4] omni norm: skipped (model=%s)", cfg_type)
        return
    try:
        from omni_xpu_kernel import norm as onorm
        patched = 0
        for m in model.modules():
            cls = type(m).__name__
            if 'RMSNorm' in cls:
                if hasattr(m, 'norm') and isinstance(getattr(m, 'norm', None), nn.Module):
                    continue
                eps = getattr(m, 'eps', 1e-6)
                if eps is None: eps = 1e-6
                if hasattr(m, 'scale') and m.scale is not None:
                    def _rms_fwd(self, x, *args, _eps=eps, **kwargs):
                        ones = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
                        y = onorm.rms_norm(ones, x.reshape(-1, x.shape[-1]).contiguous(), _eps)
                        return y.reshape(x.shape) * (self.scale.to(x.dtype) + 1.0)
                    m.forward = _rms_fwd.__get__(m)
                    patched += 1
                elif hasattr(m, 'weight') and m.weight is not None:
                    def _rms_fwd2(self, x, *args, _eps=eps, **kwargs):
                        y = onorm.rms_norm(self.weight.to(x.dtype), x.reshape(-1, x.shape[-1]).contiguous(), _eps)
                        return y.reshape(x.shape)
                    m.forward = _rms_fwd2.__get__(m)
                    patched += 1
            elif 'LayerNorm' in cls:
                if hasattr(m, 'norm') and isinstance(getattr(m, 'norm', None), nn.Module):
                    continue
                if not (hasattr(m, 'weight') and m.weight is not None):
                    continue
                eps = getattr(m, 'eps', 1e-5)
                if eps is None: eps = 1e-5
                def _ln_fwd(self, x, *args, _eps=eps, **kwargs):
                    w = self.weight.to(x.dtype)
                    b = self.bias.to(x.dtype) if self.bias is not None else None
                    y = onorm.layer_norm(x.reshape(-1, x.shape[-1]).contiguous(), w, b, _eps)
                    return y.reshape(x.shape)
                m.forward = _ln_fwd.__get__(m)
                patched += 1
        log.info("[int4] omni norm patched: %d", patched)
    except ImportError:
        pass
    except Exception:
        log.warning("[int4] omni norm patch failed, falling back to PyTorch native")
        pass


def _inject_wa4_pre_load(model, sd, quant_info, cfg_type="",
                         use_quarot=False, hadamard_H=None, qgs=128,
                         act_dtype=torch.float16, backend="w4a16",
                         mode="kernel"):
    freed, injected = 0, 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        norm = _normalize_index_path(name)
        if norm is None or norm not in quant_info: continue
        w_int4, w_scale, out_f, in_f, base, w_gs, w_corr, w_zp = quant_info.pop(norm)
        if module.weight is not None:
            freed += module.weight.numel() * module.weight.element_size()
            module.weight = nn.Parameter(torch.empty(0, device='cpu'))
        bias = None
        # bias 优先取原文件值：AIMDO 懒加载路径下 comfy 可能不消费 bias
        # （unexpected 列表里的 .bias），module.bias 会退化成默认初始化，
        # 必须绕开 comfy 的加载行为，直接读 state_dict 原值。
        # 直接从原始 state_dict 读 bias（sd 在此阶段仍保留原值，_gm 不 pop，
        # 让 comfy 正常消费避免 missing 警告；这里绕开 module.bias 确保
        # AIMDO 懒加载路径下也拿到文件原值）。
        # base 可能带 unet 前缀（diffusers 格式，如 model.diffusion_model.），
        # 但注入阶段拿到的 sd 已被 load_diffusion_model_state_dict 剥掉前缀；
        # 因此先按原 base 找，找不到再剥离已知 unet 前缀重试。
        # 注意：不能用 norm（归一化名会把 transformer_blocks 也归一成 blocks，
        # 与 sd 真实键对不上，导致 AIO 的 bias 全部漏加载）。
        sd_bias = sd.get(f"{base}.bias")
        if os.environ.get("OMNIXPU_INT4_BIAS_DIAG", "0") != "0" and injected < 3:
            _probe = [k for k in sd.keys() if "attn.to_q.bias" in k or "to_q" in k and k.endswith(".bias")]
            log.info(
                "[int4] bias diag: base=%s norm=%s found_base=%s probe_keys=%s",
                base, norm, sd_bias is not None, _probe[:3],
            )
        if sd_bias is None:
            for _pfx in ("model.diffusion_model.", "model."):
                if base.startswith(_pfx):
                    _try = base[len(_pfx):]
                    sd_bias = sd.get(f"{_try}.bias")
                    break
        if sd_bias is not None:
            bias = sd_bias.detach().clone()
            freed += bias.numel() * bias.element_size()
        elif module.bias is not None and module.bias.numel() > 0:
            bias = module.bias.data.clone()
            freed += module.bias.numel() * module.bias.element_size()
        if module.bias is not None:
            module.bias = nn.Parameter(torch.empty(0, device='cpu'))
        # 注意：不 pop bias——load_diffusion_model_state_dict 在 get_model
        # 之后会再调一次 load_model_weights，sd 里保留 bias 让第二次加载
        # 能装进 INT4XPULinear 的 bias 参数（值相同，无害），避免 839 个
        # "unet missing" 长警告。
        parts = name.split("."); parent = model
        for p in parts[:-1]: parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        if mode == "python":
            new_mod = Int4LinearPython(
                in_f, out_f, w_int4, w_scale, w_gs, bias=bias,
                act_dtype=act_dtype, use_quarot=use_quarot,
                hadamard_H=hadamard_H, correction=w_corr, quarot_gs=qgs,
            )
        elif mode == "torchao":
            new_mod = Int4LinearTorchao(
                in_f, out_f, w_int4, w_zp, w_scale, w_gs, bias=bias,
                act_dtype=act_dtype, use_quarot=use_quarot,
                hadamard_H=hadamard_H, quarot_gs=qgs,
            )
        else:
            new_mod = INT4XPULinear(in_f, out_f, w_int4, w_scale, bias=bias,
                                use_quarot=use_quarot, hadamard_H=hadamard_H,
                                group_size=w_gs, act_dtype=act_dtype, backend=backend,
                                quarot_gs=qgs, correction=w_corr,
                                zp=w_zp, tint4_mode=w_zp is not None)
        if parts[-1].isdigit(): parent[int(parts[-1])] = new_mod
        else: setattr(parent, parts[-1], new_mod)
        injected += 1
        sd.pop(f"{base}.weight", None); sd.pop(f"{base}.weight_scale", None); sd.pop(f"{base}.w4a4_group_size", None)
    # ── AIMDO 懒加载 dtype 对齐 ──
    # disable_weight_init 在 AIMDO 下走懒加载路径，未量化层权重按文件原
    # dtype 直接 Parameter(v.clone())，不转成模型计算 dtype（实测 AIO 的
    # fp16 未量化层在 bf16 模型里仍是 fp16，tint4 的 manual_cast 会转）。
    # 这里强制对齐，否则未量化层数值路径与 tint4 原厂不一致。
    _n_cast = 0
    for _m in model.modules():
        if isinstance(_m, nn.Linear) and not isinstance(_m, INT4XPULinear):
            if _m.weight is not None and _m.weight.dtype != act_dtype:
                _m.weight.data = _m.weight.data.to(act_dtype)
                _n_cast += 1
            if _m.bias is not None and _m.bias.dtype != act_dtype:
                _m.bias.data = _m.bias.data.to(act_dtype)
    if _n_cast:
        log.info("[int4] cast %d unquantized Linear to %s (AIMDO lazy dtype align)",
                 _n_cast, act_dtype)
    if os.environ.get("OMNIXPU_INT4_BIAS_DIAG", "0") != "0":
        _n_wl = sum(1 for _m in model.modules() if isinstance(_m, INT4XPULinear))
        _n_wb = sum(1 for _m in model.modules()
                    if isinstance(_m, INT4XPULinear) and _m.bias is not None)
        log.info("[int4] bias diag: INT4XPULinear=%d with_bias=%d", _n_wl, _n_wb)
    gc.collect()
    log.info("[int4] Pre-injected %d INT4XPULinear (act_dtype=%s), freed %.2f GB",
             injected, act_dtype, freed / 1024 ** 3)
    for key in list(sd.keys()):
        if key.endswith(".weight_scale") or key.endswith(".w4a4_group_size"): sd.pop(key, None)

    _patch_omni_norm(model, cfg_type)

    index = _build_wa4_lora_index(model)
    object.__setattr__(model, '_wa4_lora_index', index)
    object.__setattr__(model, '_wa4_quarot_enabled', use_quarot)
    object.__setattr__(model, '_wa4_quarot_gs', qgs)
    log.info("[int4] LoRA index: %d entries (QuaRot=%s)", len(index), use_quarot)

    if quant_info:
        log.warning("[int4] ⚠ Unmatched quant_info keys (%d): %s ...",
                    len(quant_info), list(quant_info.keys())[:15])
    return injected


class int4XPUModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            # w4a4 暂屏蔽：层间 INT4 激活传递未实现（见 INT4XPULinear.forward）。
            # w4a8-s8(88) 已从 UI 移除：实测比 84 慢且权重内存 x2，无保留价值；
            # 源码路径保留（backend="w4a8-s8" 仍可用）供开发对比。
            # 正式版默认只暴露 w4a16；开发版设 OMNIXPU_DEV_BACKENDS=1
            # 解锁 w4a8（a4 预留，层间 INT4 传递实现后开放）。
            "backend": (["w4a16", "w4a8"] if os.environ.get("OMNIXPU_DEV_BACKENDS", "0") != "0" else ["w4a16"],
                        {"default": "w4a16"}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "wa4"
    TITLE = "INT4XPU Model Loader v1.4.2n"

    def load_model(self, unet_name, backend="w4a16"):
        global _LOAD_T0
        _LOAD_T0 = time.time() if _TIMING else 0.0
        _t = _tphase("load_model enter")
        _ram_trace("load start")
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path("diffusion_models", unet_name), safe_load=True)
        _t = _tphase("load_torch_file done", _t)
        _ram_trace("load_torch_file done")
        log.info("[int4] Loading: %s", unet_name)

        weight_dtype = torch.bfloat16 if bool(sd.pop("__w4a4_weight_dtype__", torch.tensor(0)).item()) else torch.float16
        # TINT4 模型没有 wa4 的 dtype 标记：从未量化层的实际 dtype 推断
        # （tint4 量化时可能按源精度 fp16/bf16，wa4 默认 fp16 会让 bf16 模型
        #  的激活幅值溢出 fp16 范围 -> NaN/黑图）
        if (
            "w4a4_weight_dtype" not in sd
            and (
                sd.get("__tint4_format__") is not None
                or sd.get("__tint4_group_size__") is not None
            )
        ):
            for _k, _v in sd.items():
                if isinstance(_v, torch.Tensor) and _v.dtype == torch.bfloat16:
                    weight_dtype = torch.bfloat16
                    break
            log.info("[int4] tint4 dtype inferred: %s", weight_dtype)
        log.info("[int4] weight_dtype=%s (from file marker)", weight_dtype)

        use_quarot = bool(sd.pop("__w4a4_quarot__", torch.tensor(0)).item())
        qgs = int(sd.pop("__w4a4_quarot_group_size__", torch.tensor(128)).item()) if use_quarot else 128
        hadamard_H = None
        # ── TINT4 QuaRot 标记（与 wa4 的 Hadamard 旋转实现完全一致，直接复用）──
        try:
            _t4_quarot = bool(sd.get("__tint4_xpu_quarot__", torch.tensor(0)).item())
        except Exception:
            _t4_quarot = False
        if _t4_quarot:
            try:
                qgs = int(sd.get("__tint4_group_size__", torch.tensor(128)).item())
            except Exception:
                qgs = 128
            use_quarot = True
            log.info("[int4] TINT4 QuaRot ON (gs=%d)", qgs)
        if use_quarot:
            try:
                from .int4_xpu_quarot import build_hadamard
                hadamard_H = build_hadamard(qgs, device="cpu", dtype=torch.float32)
                log.info("[int4] QuaRot ON, gs=%d, H built", qgs)
            except Exception as e:
                log.warning("[int4] QuaRot H build failed: %s", e)
                use_quarot = False

        fp8_keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and v.dtype in _FP8_TYPES]
        if fp8_keys:
            for k in fp8_keys:
                sd[k] = sd[k].to(weight_dtype)
            log.info("[int4] Converted %d FP8 tensors → %s", len(fp8_keys), weight_dtype)
            _ram_trace("fp8 converted")

        # ── 融合回退阶梯：w4a4 → w4a8 → w4a16 → python / torchao ──
        # kernel 分 A/B 版且更新频繁，这里在加载时探测实际可用算子并逐级回退：
        #   w4a4（预留）→ w4a8（onednn_s8u4_gemm）→ w4a16（int4/tint4 kernel）
        #   → wa4 模型纯 python 反量化 / tint4 模型 torchao。
        caps = _kernel_caps()
        is_tint4 = (
            sd.get("__tint4_format__") is not None
            or sd.get("__tint4_group_size__") is not None
        )
        if is_tint4 and backend == "w4a8":
            _t4_gs = 64
            _mk = sd.get("__tint4_group_size__")
            if _mk is not None:
                try:
                    _t4_gs = int(_mk.item())
                except Exception:
                    pass
            if _t4_gs > 64:
                log.warning(
                    "[int4] tint4 权重 gs=%d：onednn s8u4 在 gs>64 下输出错误"
                    "（实测 corr~0.1，非崩溃但数值错），自动回退 w4a16；"
                    "用小 gs（32/64）重新量化即可走 a8", _t4_gs)
                backend = "w4a16"
        _mode = "kernel"  # kernel 原生（wa4 或 tint4）
        backend = _resolve_backend(backend, caps)
        if backend == "w4a16":
            if is_tint4:
                if caps["tint4"]:
                    _mode = "kernel"
                elif caps["int4"]:
                    _mode = "converted"  # 仅旧 onednn_int4_gemm：走转换路径
                else:
                    _mode = "torchao"
                    log.warning("[int4] tint4 kernel 不可用，回退 torchao（需 torchao-xpu）")
            else:
                _mode = "kernel" if caps["int4"] else "python"
                if _mode == "python":
                    log.warning("[int4] int4 kernel 不可用，回退纯 python 反量化")
        log.info("[int4] backend=%s mode=%s caps=%s", backend, _mode, caps)

        qi = _index_int4_from_sd(sd, backend=backend, mode=_mode)
        _t = _tphase("int4 indexed", _t)
        _ram_trace("int4 indexed")
        # 先记录编辑参考标记（下面会 pop 掉，但 cfg 检测需要它）
        _t4_edit_marker = any(
            k.endswith("__index_timestep_zero__") for k in sd
        )
        # tint4 标记键消费完即清（避免 unexpected 警告）；QuaRot 标记先读出来
        # 标记键消费完即清；但 __index_timestep_zero__ 必须保留——模型注入
        # default_ref_method=index_timestep_zero 后会注册同名 buffer，sd 里
        # 留着它才能被 load_state_dict 消费（否则报 "unet missing"）。
        # （diffusers 前缀下该键带 model.diffusion_model. 前缀，按结尾匹配）
        for _mk in list(sd.keys()):
            # 任何以 __index_timestep_zero__ 结尾的键（含 diffusers 前缀）
            # 都保留，供模型 buffer 消费；其余 __ 标记键清除。
            if _mk.endswith("__index_timestep_zero__"):
                continue
            if _mk.startswith("__"):
                sd.pop(_mk, None)
        # ── tint4 转换的 GPU 缓冲立即释放 ──
        # 转换要经 GPU 做 10GB H2D + 字节翻转 + D2H；AIMDO 激活时 load 末尾的
        # empty_cache 会被让路跳过、AIMDO 物理页池也不归还，转换残留 ~10GB
        # 与随后权重 prewarm / CLIP 叠加 → 溢出共享显存 15-16GB 且不回落
        # （实测 CSV：专用 15.74GB 峰值 → 共享 0.09→16.16GB 持续）。这里
        # 转换一结束就清，避免残留进入后续加载。
        try:
            if _aimdo_manages():
                from comfy_aimdo import control as _aimdo_ctl
                _ok = _aimdo_ctl.empty_xpu_allocator_cache(wait=True)
                if _TIMING:
                    log.info(
                        "[wa4-timing] aimdo empty after tint4 conv -> ok=%s total=%.0fMB",
                        _ok, _aimdo_ctl.get_total_vram_usage() / (1024 ** 2),
                    )
            else:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
                if _TIMING:
                    log.info("[wa4-timing] xpu.empty_cache after tint4 conv")
        except Exception as _e:
            log.warning("[int4] tint4 conv cache release failed: %r", _e)

        import comfy.model_detection as md
        clean_sd = {k: v for k, v in sd.items()
                    if k.endswith((".weight", ".bias", ".scale"))
                    or k.startswith(("text_encoders.", "vae."))}
        prefix = md.unet_prefix_from_state_dict(clean_sd)
        tsd = comfy.utils.state_dict_prefix_replace(clean_sd, {prefix: ""}, filter_keys=True)
        cfg = md.model_config_from_unet(tsd if len(tsd) > 0 else clean_sd, "")
        cfg_type = type(cfg).__name__ if cfg else "UNKNOWN"
        # ── Qwen-Image-Edit 参考方式标记 ──
        # clean_sd 过滤把 __index_timestep_zero__ 标记键滤掉了，comfy 检测
        # 会退回架构默认 "index"；tint4 原厂（其配置缓存）用
        # "index_timestep_zero"。检测到标记则显式注入，保证编辑参考一致。
        if cfg is not None and _t4_edit_marker:
            try:
                cfg.unet_config["default_ref_method"] = "index_timestep_zero"
                log.info("[int4] default_ref_method=index_timestep_zero (tint4 edit marker)")
            except Exception:
                pass
        log.info("[int4] Config: %s%s", cfg_type, " [QuaRot]" if use_quarot else "")
        if cfg is None: raise RuntimeError("[int4] Architecture detection failed")

        act_dtype = weight_dtype
        log.info("[int4] act_dtype=%s (跟随量化器标记)", act_dtype)

        _o1 = md.model_config_from_unet
        md.model_config_from_unet = lambda *a, **kw: cfg
        _o2 = cfg.get_model
        def _gm(self, state_dict, prefix=""):
            for k in list(state_dict.keys()):
                if k.endswith((".weight_scale", ".w4a4_group_size")):
                    state_dict.pop(k, None)
                elif k.endswith(".weight"):
                    base = k[:-7]
                    norm_test = _normalize_index_path(base)
                    if norm_test is not None and norm_test in qi:
                        state_dict.pop(k, None)
            m = _o2(state_dict, prefix)
            _inject_wa4_pre_load(m, state_dict, qi, cfg_type,
                                 use_quarot=use_quarot, hadamard_H=hadamard_H,
                                 qgs=qgs, act_dtype=act_dtype, backend=backend,
                                 mode=_mode)
            return m
        cfg.get_model = types.MethodType(_gm, cfg)

        try:
            model = comfy.sd.load_diffusion_model_state_dict(
                sd, model_options={
                    "custom_operations": comfy.ops.disable_weight_init,
                    "dtype": act_dtype,
                })
        finally:
            cfg.get_model = _o2; md.model_config_from_unet = _o1
        _t = _tphase("model built", _t)
        _ram_trace("model built")

        # ── AIMDO 适配（让路）──
        try:
            from .int4_xpu_aimdo import patch_aimdo_xpu
            patch_aimdo_xpu(model)
        except Exception:
            pass

        # 注意：不做权重预热（ensure_xpu）。实测预搬会让模型权重与
        # 随后加载的 bf16 CLIP 在显存中叠加（15.5GB 持续不回落，CLIP 卸载
        # 异常）。保持 forward 惰性 .to(dev)：CLIP 阶段权重留在 CPU，
        # CLIP 卸载后采样时才逐层搬入——与 WINT4/tint4 的驻留策略一致。
        try:
            dm = model.model.diffusion_model
            while hasattr(dm, '_orig_mod'):
                dm = dm._orig_mod
            # 预热目标：采样首次 forward 时一次性搬入所有权重（CLIP 已卸载，
            # 不叠加）；重置标志以支持同一进程多次加载。
            INT4XPULinear._prewarm_target = dm
            INT4XPULinear._prewarm_done = False
            # ── 轻量探针（QW 显存约束，AIMDO 下也装）──
            # 预热已消除采样中的显存爬升（实测无探针也稳定），探针默认关闭；
            # 需要对比时设 OMNIXPU_INT4_PROBE=1 可临时恢复。
            if cfg_type == "QwenImage" and os.environ.get("OMNIXPU_INT4_PROBE", "0") != "0":
                n_probe = _wrap_attn_probe(dm)
                if n_probe:
                    log.info("[int4] attention probe: %d modules (reset_peak+sync)",
                             n_probe)
            if backend == "w4a8" and cfg_type == "QwenImage":
                n_attn = _wrap_qwenimage_attn_shared_quant(dm)
                if n_attn:
                    log.info("[int4] QwenImage s8 closure: shared-quant %d attention blocks",
                             n_attn)
                # GELU s8-out：默认开启（QW 实测稳定路径）。需要 A/B 时设
                # OMNIXPU_INT4_GELU_S8=0 关闭。
                if os.environ.get("OMNIXPU_INT4_GELU_S8", "1") != "0":
                    n_gelu = _wrap_qwenimage_gelu_s8(dm)
                    if n_gelu:
                        log.info("[int4] QwenImage s8 closure: gelu s8-out on %d MLPs",
                                 n_gelu)
                n_lin, n_gate = _wrap_qwenimage_block_s8(dm)
                if n_lin or n_gate:
                    log.info("[int4] QwenImage s8 closure: s8-out on %d proj, "
                             "%d gates dequant", n_lin, n_gate)
            elif backend == "w4a8":
                # 通用闭包：任何架构（FLUX/Krea2/ZIT/Boogu/...）按角色识别
                # 输出投影 s8 + 共享量化 + SwiGLU 融合，边界由 W4ActS8
                # 张量协议自动反量化。
                if os.environ.get("OMNIXPU_INT4_AUTO_S8", "1") != "0":
                    counts = _auto_s8_closure(dm)
                    log.info("[int4] auto s8 closure: out_s8=%d attn_shared=%d swiglu=%d",
                             counts["out_s8"], counts["attn_shared"], counts["swiglu"])
                else:
                    log.info("[int4] auto s8 closure: DISABLED (OMNIXPU_INT4_AUTO_S8=0)")
        except Exception as e:
            log.warning("[int4] s8 closure setup skipped: %s", e)

        del sd, qi, clean_sd, tsd
        gc.collect()
        _ram_trace("sd freed + gc")
        # AIMDO 接管显存时 wa4 不手动同步/清缓存（避免打断 AIMDO 水位换页）；
        # 未启用 AIMDO 时保留原兜底（同步 + 清缓存）。
        if not _aimdo_manages():
            try:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            except Exception:
                pass

        # ── detach 包装：QW 释放（保留 CPU 产物）；小模型（Krea2 等）常驻保速度 ──
        _o_detach = model.detach
        def _wa4_detach(unpatch_all=True):
            try:
                # AIMDO 接管时由 AIMDO 统一管理（不主动释放/同步/清缓存）；
                # 未启用 AIMDO 时保留原 detach 释放兜底。
                if not _aimdo_manages():
                    if cfg_type in _DETACH_RELEASE_MODELS:
                        dm = model.model.diffusion_model
                        while hasattr(dm, '_orig_mod'):
                            dm = dm._orig_mod
                        for m in dm.modules():
                            if isinstance(m, INT4XPULinear):
                                m.release_xpu()
                    torch.xpu.synchronize()
                    torch.xpu.empty_cache()
            except Exception:
                pass
            return _o_detach(unpatch_all)
        object.__setattr__(model, 'detach', _wa4_detach)

        log.info("[int4] Load complete (dtype=%s, backend=%s)", act_dtype, backend)
        _tphase("load complete", _t)
        _ram_trace("load complete")
        return (model,)


NODE_CLASS_MAPPINGS = {"int4XPUModelLoader": int4XPUModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"int4XPUModelLoader": "INT4XPU Model Loader v1.4.2n"}


# ── 调试：捕获 QW 模型首次前向的真实输入（OMNIXPU_INT4_DUMP_FWD=路径）──
if os.environ.get("OMNIXPU_INT4_DUMP_FWD"):
    try:
        import comfy.ldm.qwen_image.model as _qwm
        _dump_path = os.environ["OMNIXPU_INT4_DUMP_FWD"]
        _orig_qw_fwd = _qwm.QwenImageTransformer2DModel._forward

        def _qw_fwd_dump(self, *args, **kwargs):
            out = _orig_qw_fwd(self, *args, **kwargs)
            try:
                _n = getattr(_qwm, "_wa4_dump_count", 0)
                if _n < 24:
                    _qwm._wa4_dump_count = _n + 1
                    _store = getattr(_qwm, "_wa4_dump_store", [])
                    _store.append(
                        {"args": args, "kwargs": kwargs, "out": out}
                    )
                    _qwm._wa4_dump_store = _store
                    torch.save(
                        _store,
                        _dump_path,
                    )
                    log.info("[int4] dumped first QW forward -> %s", _dump_path)
            except Exception as _e:
                log.warning("[int4] fwd dump failed: %r", _e)
            return out

        _qwm.QwenImageTransformer2DModel._forward = _qw_fwd_dump
        log.info("[int4] QW forward dump armed -> %s", _dump_path)
    except Exception as _e:
        log.warning("[int4] QW forward dump arm failed: %r", _e)
