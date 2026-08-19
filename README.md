# int4-omnixpu

统一 INT4 加载器：wa4 + tint4 模型融合加载，经 `omni_xpu_kernel` 的
oneDNN INT4 GEMM 家族加速（A770/DG2 XPU）。

## 依赖

- **omni_xpu_kernel**（必需加速后端，**手动安装**）：分 A/B 版且更新频繁，
  本插件不提供自动安装。安装方式见
  [ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU) 的
  README，或从 `omni-xpu-kernel` 仓库按 A770 目标编译 wheel 后
  `pip install <wheel>`。
- **torchao**（仅 tint4 模型无 kernel 回退时必需）：插件启动时自动检测，
  缺失/过旧时自动 `pip install torchao --isolated --index-url
  https://download.pytorch.org/whl/xpu`（tint4 插件同款机制）。

## 支持的模型格式

| 格式 | 权重表示 | scale | 说明 |
|---|---|---|---|
| wa4 | packed u4（有符号 nibble） | per-group f16 | 对称 int4，组大小 32/64/128 |
| tint4 | int32 qdata（每 int32 8 个 nibble） | per-block f16 + per-block zp | 非对称 int4（torchao 格式），QuaRot 支持 |

模型架构自动检测（Qwen-Image / Krea2 / Boogu / Z-Image / Flux2 等）。

## 后端与回退阶梯

加载时按 `omni_xpu_kernel` 实际可用算子自动选择：

```
请求 w4a4 / w4a8
  ├─ w4a4（层间 INT4 传递未实现，预留）
  └─ w4a8（onednn_s8u4_gemm）
       ├─ tint4 模型额外要求权重 gs<=64（A770 上 gs>64 的 s8u4 会崩溃）
       └─ gs>64 → 自动回退 w4a16
请求/回退到 w4a16
  ├─ wa4 模型：onednn_int4_gemm(_preconverted) 缺失 → 纯 python 反量化
  └─ tint4 模型：onednn_int4_gemm_tint4 缺失
        ├─ 有 onednn_int4_gemm → 转换路径（signed u4 + 修正项）
        └─ 都没有 → torchao（Int4PlainInt32Tensor，需 torchao-xpu）
```

日志会打印 `backend=... mode=...` 说明实际生效路径。

## 节点

- **wa4ModelLoader**：`unet_name` + `backend`（w4a16 / w4a8），wa4 与 tint4
  文件都可用同一个加载器。
- **wa4ModelQuantizer**：把 fp16/bf16/fp8/int8 模型量化为 wa4 格式。
- **wa4 LoRA Loader / Stack**：LoRA 注入（GPU 缓存，避免逐层 H2D 的 CPU
  高占用）。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `OMNIXPU_WA4_TINT4_NATIVE` | 1 | tint4 走原生 kernel（零转换）；0=转换路径 |
| `OMNIXPU_ATTN_NAN_CHECK` | 0 | 关 ESIMD fp16 NaN 全扫（提速，安全版为 1） |
| `OMNIXPU_SDP_CACHE_AUTOCLEAR` | keep | sidecar 缓存跨卸载保留 |
| `OMNIXPU_WA4_TIMING` | 0 | 分阶段计时日志 |
| `OMNIXPU_WA4_BIAS_DIAG` | 0 | bias 加载诊断日志 |

## 已知边界

- tint4 的 a8：s8 激活闭包 + raw 权重（不异或）+ per-block zp 修正项
  （(8-zp)*scale 小矩阵外积），实测 Boogu-Edit(gs=32) 与 a16 同 seed 出图
  corr 0.9997；**gs>64 的 tint4 模型**（QW/kr2 等 gs=128）在 A770 的
  oneDNN s8u4 上不支持（崩溃），自动回退 w4a16。
- WAN / LTX2 视频模型：int4 精度损失较高，暂缓适配。
- 无 kernel 的 wa4 回退（纯 python）与 tint4 回退（torchao）速度显著慢于
  kernel 路径，仅作可用性兜底。
