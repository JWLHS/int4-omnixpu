# int4-omnixpu

面向 Intel Arc XPU 的统一 **INT4 模型加载器**：同时支持 wa4 格式与
torchao 后端量化的非对称 INT4 格式（tint4），经 `omni_xpu_kernel` 的
oneDNN INT4 GEMM 原生加速。

> 当前正式版主打 **w4a16 后端**（int4 权重 + 16bit 激活）。a8/a4 后端仍在
> 开发中，界面上暂时隐藏。

## 插件职责边界（重要）

**int4-omnixpu 只做"加载 + INT4 GEMM"**：模型文件读取、权重注入、LoRA、
量化、以及直接调用 `omni_xpu_kernel.svdq` 的 INT4 算子。

- **不依赖 ComfyUI-OmniXPU 插件**：GEMM 直接走 kernel 包，OmniXPU 插件
  装不装、开不开都不影响加载与出图。
- **ComfyUI-OmniXPU 是可选的性能增强**：它提供 norm（RMSNorm/LayerNorm）
  与 attention 的 ESIMD 加速；不装时模型走 ComfyUI 原生 torch 路径，
  功能完整、速度稍慢。**是否启用由你自己决定。**
- 唯一必需后端：`omni_xpu_kernel`（kernel 包），按下面 A/B 系列任选一个
  安装。

## 依赖：kernel 后端（A/B 系列区分）

kernel 分 A/B 两个系列，按你的平台选对应的仓库编译 wheel 后安装
（`pip install <wheel>`），本插件不自动安装 kernel。

| 系列 | 目标平台 | kernel 仓库 | 配套插件（可选） |
|---|---|---|---|
| **B 系列** | BMG / PTL-H（非 A770） | 原仓库 [intel/llm-scaler](https://github.com/intel/llm-scaler)（`omni/omni_xpu_kernel`，按上游 README 编译） | 原仓库内置的 [ComfyUI-OmniXPU](https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU) |
| **A 系列** | A770 / DG2 | SDP 适配仓库 [Blackwood416/omni-xpu-kernel](https://github.com/Blackwood416/omni-xpu-kernel)（按仓库 README 编译） | [Blackwood416/ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU) |

说明：
- 配套插件（ComfyUI-OmniXPU）只提供 norm/attention 加速，**非必需**；
  装了更好，不装不影响本插件功能。
- 本插件对 A/B 无感知：加载时探测 kernel 里实际存在的算子，有就调用、
  没有就安全回退。

## 模型下载

已量化的可用模型（INT4）：

- **wa4 格式**：[百度网盘](https://pan.baidu.com/s/5OWmgfWfYzBzb1R5C7WWPMw)
- **tint4 / torchao 格式**：[夸克网盘](https://pan.quark.cn/s/a324b2c9881b)

> **关于 tint4（torchao 格式）**：这是基于 **torchao 后端**量化的非对称
> INT4（Int4PlainInt32Tensor：int32 qdata + per-block zero point + per-block
> scale，`w = (q - zp) * scale`）。它由
> [ComfyUI-TINT4](https://github.com/JWLHS/ComfyUI-TINT4) 插件（原加载器）
> 量化产生。
>
> - **原加载器**（ComfyUI-TINT4）：走 torchao 的 `Int4PlainInt32Tensor`
>   后端，纯 Python/torchao 路径，无 XPU kernel 加速。
> - **本插件**：直接识别 tint4 文件，调用 `omni_xpu_kernel` 的原生 INT4
>   GEMM（per-block zp 在 oneDNN 内应用），**零转换、零修正项**，速度显著
>   更快。
> - **差异**：非对称 per-block zero point 能把偏置的权重分布逐块归中，
>   相比对称 INT4 量化误差更小（编辑/姿态迁移类能力保留更好）。
> - **使用差异**：tint4 文件用本插件的 `wa4ModelLoader`（backend=w4a16）
>   直接加载即可，不需要装 ComfyUI-TINT4 加载器；两套可以共存，不会冲突。

下载后把模型放到 `ComfyUI/models/diffusion_models/` 下（可自建子目录），
加载时填相对路径即可。

## 使用方式

1. 按上面选一个系列的 kernel 编译安装。
2. 把本插件放进 `ComfyUI/custom_nodes/`。
3. 模型放入 `models/diffusion_models/`。
4. 工作流中用 **wa4ModelLoader** 节点：`unet_name` 选模型文件，
   `backend` 选 **w4a16**。

节点一览：

- **wa4ModelLoader**：wa4 / tint4 统一加载（当前正式版 backend 仅 w4a16；
  a8/a4 隐藏待开发）。
- **wa4ModelQuantizer**：把 fp16/bf16/fp8/int8 模型量化为 wa4 格式。
- **wa4 LoRA Loader / Stack**：LoRA 注入（GPU 侧缓存，避免逐层 H2D 造成
  CPU 高占用）。

## 后端回退（安全兜底）

插件加载时探测 kernel 实际可用算子，自动选择路径；**kernel 缺失或算子
不存在时不会报错**，按下面阶梯安全回退：

```
请求 w4a16
  ├─ wa4 模型：
  │    ├─ onednn_int4_gemm(_preconverted) 存在 → kernel 原生
  │    └─ 缺失 → 纯 python 反量化（一次性反量化 + F.linear）
  └─ tint4（torchao）模型：
       ├─ 原生算子存在（onednn_int4_gemm_torchao / zp 参数）→ kernel 原生
       ├─ 有 onednn_int4_gemm → 转换路径（signed u4 + 修正项）
       └─ 都没有 → torchao（Int4PlainInt32Tensor，缺库自动安装 torchao-xpu）
```

日志会打印 `backend=... mode=...` 说明实际生效路径。回退路径速度显著慢于
kernel，但保证可用、不中断。

## 性能参照（8 步，1024×1024，同 seed）

同一模型 wa4 与 tint4（torchao）格式在 w4a16 下的稳态每步耗时：

| 模型 | wa4 | tint4 | 备注 |
|---|---|---|---|
| Krea2 turbo | ~1.8s/步 | ~1.95s/步 | 快慢交替步为模型结构行为 |
| Qwen-Edit | ~1.5s/步 | ~1.5s/步 | 编辑能力两格式相当 |
| Qwen-AIO | ~1.5s/步 | ~2.1s/步 | tint4 走转换/原生路径差异 |
| Z-Image Turbo | ~1.05s/步 | ~1.06s/步 | 最快 |

（完整 20 模型冷/热对比与每步明细见仓库 `docs/` 下的测试记录。）

## 环境变量

**本插件**的可选变量（默认即最优，一般不用设）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `OMNIXPU_WA4_TINT4_NATIVE` | 1 | tint4 走原生 kernel；0=转换路径 |
| `OMNIXPU_WA4_AUTO_S8` | 1 | a8 自动 s8 闭包（a8 开放后生效） |
| `OMNIXPU_WA4_TIMING` | 0 | 分阶段计时日志 |
| `OMNIXPU_WA4_BIAS_DIAG` | 0 | bias 加载诊断日志 |
| `OMNIXPU_RAM_TRACE` | 0 | 内存采样日志 |

**以下变量属于 A 系列 kernel / ComfyUI-OmniXPU 插件**（不是本插件的），
常见配置与含义：

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNIXPU_ATTN_NAN_CHECK` | 0 | A 系列 kernel ESIMD attention 的 fp16 NaN 全扫开关；默认关（提速），安全排查时开 1 |
| `OMNIXPU_SDP_CACHE_AUTOCLEAR` | keep | A 系列 sdp 侧车缓存跨模型卸载是否保留；默认 keep 避免每轮重编 |
| `OMNIXPU_ATTENTION` / `OMNIXPU_NORM` | 1 | ComfyUI-OmniXPU 的 attention/norm 加速开关；默认开，设 0 关闭 |

> 这些变量的具体默认值以你安装的 kernel/插件仓库 README 为准；本插件只保证
> **kernel 装上后零环境变量直接可用**。上面"最快生图"的配置仅作参考，不设
> 也不影响正确性。

## 已知边界

- a8 / a4 后端开发中，当前正式版为 w4a16。
- WAN / LTX2 视频模型 int4 精度损失较高，暂缓适配。
- kernel 缺失时回退路径（python/torchao）慢，仅作可用性兜底。

## 鸣谢

- [intel/llm-scaler](https://github.com/intel/llm-scaler)（B 系列 kernel
  原仓库，omni/omni_xpu_kernel）
- [Blackwood416/omni-xpu-kernel](https://github.com/Blackwood416/omni-xpu-kernel)
  与 [Blackwood416/ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU)
  （A 系列 kernel 与配套插件，SDP 适配）
- [ComfyUI-TINT4](https://github.com/JWLHS/ComfyUI-TINT4)（tint4/torchao
  原加载器与量化工具）
- [torchao](https://github.com/pytorch/ao)（INT4 非对称量化格式后端）
- ComfyUI 生态与各位测试参考
