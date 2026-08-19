# int4-omnixpu

[中文](README.md) | **English**

A unified **INT4 model loader** for Intel Arc XPU: loads both the `wa4`
format and the asymmetric INT4 format quantized on the torchao backend
(`tint4`), accelerated natively by the oneDNN INT4 GEMM family in
`omni_xpu_kernel`.

> The current release is built around the **w4a16 backend** (INT4 weights +
> 16-bit activations). The a8/a4 backends are still under development and are
> hidden from the UI.

## What this plugin does (and doesn't)

**int4-omnixpu only handles "loading + INT4 GEMM"**: model file reading,
weight injection, LoRA, quantization, and direct calls to the INT4 GEMM
interface (`int4_gemm`, backed by the `onednn_int4_gemm*` ops in
`omni_xpu_kernel`).

- **No dependency on the ComfyUI-OmniXPU plugin**: GEMM calls go straight to
  the kernel package. Whether the OmniXPU plugin is installed or enabled has
  no effect on loading or output.
- **ComfyUI-OmniXPU is an optional performance enhancement**: it provides
  ESIMD acceleration for norm (RMSNorm/LayerNorm) and attention. Without it,
  models run on ComfyUI's native torch path — fully functional, just slower.
  **Enable it at your discretion.**
- The only required backend is the `omni_xpu_kernel` package; pick one of the
  A/B series below.

## Dependencies: kernel backend (A/B series)

The kernel ships as two series. Choose the repo for your platform, build the
wheel following that repo's README, and install it with
`pip install <wheel>`. This plugin does not install the kernel for you.

| Series | Target | Kernel repo | Companion plugin (optional) |
|---|---|---|---|
| **B series** | BMG / PTL-H (non-A770) | Upstream [intel/llm-scaler](https://github.com/intel/llm-scaler) (`omni/omni_xpu_kernel`, build per upstream README) | In-repo [ComfyUI-OmniXPU](https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU) |
| **A series** | A770 / DG2 | SDP-adapted repo [Blackwood416/omni-xpu-kernel](https://github.com/Blackwood416/omni-xpu-kernel) (build per repo README) | [Blackwood416/ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU) |

Notes:
- The companion plugin (ComfyUI-OmniXPU) only adds norm/attention
  acceleration — **optional**; install it for better speed, skip it without
  affecting this plugin's function.
- This plugin is A/B agnostic: at load time it probes which ops actually
  exist in the kernel, calls what is available, and falls back safely when
  something is missing.

### torchao-xpu dependency (auto-installed)

**torchao** is the backend implementation of the asymmetric INT4 format
(`Int4PlainInt32Tensor`) and also the **fallback runtime for tint4/torchao
models when no kernel is available**. On startup the plugin checks: if the
required torchao is missing or outdated, it automatically runs
`pip install torchao --isolated --index-url https://download.pytorch.org/whl/xpu`
(the same mechanism as the original tint4 plugin).

Purpose: **tint4 models load and run even on machines without the kernel (or
with a kernel missing the ops)** — no manual torchao install required. With
a kernel you get native acceleration; without one you fall back to the
torchao path (complete functionality, slower).

## Model downloads

Quantized INT4 models:

- **wa4 format**: [Baidu Netdisk](https://pan.baidu.com/s/5OWmgfWfYzBzb1R5C7WWPMw)
- **tint4 / torchao format**: [Quark Netdisk](https://pan.quark.cn/s/a324b2c9881b)

> **About tint4 (torchao format)**: asymmetric INT4 quantized on the torchao
> backend (`Int4PlainInt32Tensor`: int32 qdata + per-block zero point +
> per-block scale, `w = (q - zp) * scale`). It is produced by the
> [ComfyUI-TINT4](https://github.com/JWLHS/ComfyUI-TINT4) plugin (the
> original loader).
>
> - **Original loader** (ComfyUI-TINT4): runs on torchao's
>   `Int4PlainInt32Tensor` backend — pure Python/torchao, no XPU kernel
>   acceleration.
> - **This plugin**: recognizes tint4 files directly and calls the native
>   INT4 GEMM in `omni_xpu_kernel` (per-block zp applied inside oneDNN) —
>   **zero conversion, zero correction**, significantly faster.
> - **Why asymmetric**: the per-block zero point centers biased weight
>   distributions per block, so quantization error is lower than symmetric
>   INT4 (editing / pose-transfer capability is preserved better).
> - **Usage difference**: load tint4 files with this plugin's
>   `int4XPUModelLoader` (backend=w4a16) — no need to install the
>   ComfyUI-TINT4 loader; the two can coexist without conflict.

Put models under `ComfyUI/models/diffusion_models/` (subfolders are fine)
and use the relative path when loading.

## Usage

1. Pick the series for your GPU: **A series** for A770/DG2, **B series**
   for everything else (BMG/PTL-H). Build and install the kernel from that
   repo's README.
2. Put this plugin in `ComfyUI/custom_nodes/`.
3. Put models in `models/diffusion_models/`.
4. In your workflow use the **int4XPUModelLoader** node: select the model
   file in `unet_name` and choose **w4a16** for `backend`.

Nodes:

- **int4XPUModelLoader**: unified loader for wa4 / tint4 (current release
  exposes only w4a16; a8/a4 hidden pending development).
- **int4XPUModelQuantizer**: quantize fp16/bf16/fp8/int8 models to the wa4
  format.
- **INT4XPU LoRA Loader / Stack**: LoRA injection (GPU-side cache, avoiding
  per-layer H2D CPU spikes).

## Backend fallback (safe)

At load time the plugin probes which kernel ops actually exist and picks the
path automatically; **missing kernel or missing ops never cause an error**.
The ladder:

```
request w4a16
  ├─ wa4 model:
  │    ├─ onednn_int4_gemm(_preconverted) present -> native kernel
  │    └─ missing -> pure python dequant (one-shot dequant + F.linear)
  └─ tint4 (torchao) model:
       ├─ native op present (onednn_int4_gemm_torchao / zp param) -> native kernel
       ├─ onednn_int4_gemm present -> conversion path (signed u4 + correction)
       └─ neither -> torchao (Int4PlainInt32Tensor; torchao-xpu auto-installed)
```

The log prints `backend=... mode=...` to show the active path. Fallback
paths are significantly slower than the kernel but are guaranteed to work
without interruption.

## Performance reference (8 steps, 1024×1024, same seed)

Steady-state seconds per step for the same model in wa4 vs tint4 (torchao)
format at w4a16:

| Model | wa4 | tint4 | Notes |
|---|---|---|---|
| Krea2 turbo | ~1.8 | ~1.95 | alternating fast/slow steps are model behavior |
| Qwen-Edit | ~1.5 | ~1.5 | editing capability is equivalent in both formats |
| Qwen-AIO | ~1.5 | ~2.1 | tint4 conversion/native path difference |
| Z-Image Turbo | ~1.05 | ~1.06 | fastest |

(Full 20-model cold/warm comparison and per-step details are in the
`docs/` folder of this repo.)

## Environment variables

**This plugin's** optional variables (defaults are optimal; normally nothing
needs setting):

| Variable | Default | Purpose |
|---|---|---|
| `OMNIXPU_INT4_TINT4_NATIVE` | 1 | tint4 uses the native kernel (per-block zp straight into oneDNN); set 0 to **convert tint4 to the wa4 format online** at load and use the wa4 path (old-kernel compatibility, slower) |
| `OMNIXPU_INT4_TIMING` | 0 | phase timing logs |
| `OMNIXPU_INT4_BIAS_DIAG` | 0 | bias loading diagnostics |
| `OMNIXPU_RAM_TRACE` | 0 | memory sampling logs |

**The variables below belong to the A-series kernel / ComfyUI-OmniXPU
plugin** (not this plugin). The table lists the **upstream defaults** first,
then **our local test settings (reference only)** and why:

| Variable | Upstream default | Our local test value (ref) | Meaning |
|---|---|---|---|
| `OMNIXPU_ATTN_NAN_CHECK` | 1 (on) | 0 | fp16 NaN full-scan for A-series ESIMD attention. Upstream keeps it on (safe); locally we turn it off for speed and back on when debugging black/NaN images |
| `OMNIXPU_SDP_CACHE_AUTOCLEAR` | clear | keep | whether the A-series sdp sidecar cache survives model unload. Upstream clears it every round (saves VRAM); locally we keep it to avoid recompiling |
| `OMNIXPU_ATTENTION` / `OMNIXPU_NORM` | 1 | 1 | ComfyUI-OmniXPU attention/norm acceleration switches; set 0 to disable |

> Principle: **everything works with no variables set** (upstream defaults).
> Our local settings only exist for fastest generation / debugging and are
> reference only. Exact defaults follow the kernel/plugin repo you install.

## Known limitations

- a8 / a4 backends are under development; the current release is w4a16.
- WAN / LTX2 video models are deferred (INT4 precision loss is too high).
- Without a kernel, fallback paths (python/torchao) are slow — availability
  fallback only.

## Credits

- [intel/llm-scaler](https://github.com/intel/llm-scaler) (B-series kernel
  upstream, `omni/omni_xpu_kernel`)
- [Blackwood416/omni-xpu-kernel](https://github.com/Blackwood416/omni-xpu-kernel)
  and [Blackwood416/ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU)
  (A-series kernel and companion plugin, SDP-adapted)
- [ComfyUI-TINT4](https://github.com/JWLHS/ComfyUI-TINT4) (original
  tint4/torchao loader and quantization tooling)
- [torchao](https://github.com/pytorch/ao) (asymmetric INT4 format backend)
- The ComfyUI ecosystem and everyone who contributed test feedback
