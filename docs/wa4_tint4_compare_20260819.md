# wa4 vs tint4 全量对比报告（2026-08-19）

- 环境：Arc A770 16GB，ComfyUI + 融合加载器（ComfyUI-Int4OmniXPU），AIMDO 启用，
  每模型跑完卸载（`/free`），1024×1024，同架构同 seed。
- 测试模型：wa4 6 个 + tint4 14 个，全部加载成功（20/20）。

## 1. 配对识别（同 seed 输出相似度定位同基底）

| wa4 文件 | 对应 tint4 | corr | mean_diff |
|---|---|---|---|
| booguTtfp | booguturbo_tint4 | 0.885 | 19.9 |
| f2k9btfp | F2K-TURBO-9B_tint4 | 0.862 | 20.2 |
| kr2fp | krea2turbo_tint4（**kr2fp 就是 turbo**） | 0.758 | 29.4 |
| zitfp | Zimage-Turbo_tint4（**zitfp 就是 turbo**） | 0.689 | 26.5 |
| qw-edit-2511-bf | qwen-edit-2511_tint4 | 0.918 | 10.7 |
| qw-aio-23-bf | qwaio23_tint4 | 0.813 | 24.3 |

单跑（无 wa4 对照）：t4_booguedit、t4_f2kdb、t4_kr2raw、t4_moodykr2、t4_zimagebase、
t4_zitns、t4_zibv1/v2。

## 2. 每模型耗时（wall，含加载+prewarm+采样）

| 配对 | wa4 | tint4 | 差 |
|---|---|---|---|
| boogu turbo | 34.8s | 39.2s | tint4 +4.4s |
| f2k turbo | 27.1s | 33.4s | tint4 +6.3s |
| krea2 turbo | 42.9s | 39.6s | wa4 +3.3s |
| qwedit | 54.1s | 43.8s | wa4 +10.3s |
| qwaio | 49.5s | 52.6s | tint4 +3.1s |
| zit turbo | 31.0s | 21.0s | wa4 +10.0s |

差异主因：wa4 每次 prewarm 7.4s（CPU 重打包）vs tint4 3.3s（原生免打包），
加上采样期 kernel 差异；各模型方向不一致（wa4 在 qwedit/zit 反而更慢）。

## 3. 内存/显存（重点）

### 发现并修复：tint4 内存 +10GB（已修复）
- **根因**：tint4 原生路径把 int32 qdata 的字节视图当 packed 权重，prewarm 搬 GPU
  后 **CPU 侧 10GB int32 从不释放**（`w_int4` 一直引用）；wa4 路径 prewarm 后释放
  CPU 副本。→ tint4 变成 GPU 10GB + CPU 10GB 并存。
- **修复**：`int4_loader.py` 的 `_prepare`（tint4 分支）在建立字节视图后
  `w_int4=None`，prewarm 搬 GPU 后 CPU 存储即释放。
- **实测（qwaio 配对，独立进程）**：

| | 修复前 | 修复后 |
|---|---|---|
| tint4 qwaio RSS（prewarm 后） | 11.8GB | **0.9GB** |
| wa4 qwaio RSS | 0.9GB | 0.9GB |
| tint4 GPU 占用 | 10.8GB | 10.8GB |
| wa4 GPU 占用 | 11.0GB | 11.0GB |

修复后两者内存/显存对称（权重层面无 +10GB 差异）。

### wa4 显存高 vs tint4 内存高
- 之前观察到的差异（wa4 显存 14.4-16.8 vs tint4 13.4-13.6）在权重层面并不存在
  （探针：10.8 vs 11.0GB 基本持平）；差异来自 CLIP/VAE 加载时序与测量窗口。
- 全量测试（修复后）：共享显存全程无溢出（峰值 ≤1.6GB，均正常回落）。

## 4. 仍存在的问题
- **每次运行都重新 prewarm**（wa4 与 tint4 都是）：QW 系在 detach 时释放 GPU
  权重回 CPU，下次运行重新搬——这是"每次重新加载"观感与部分速度差异的来源。
- tint4 与 wa4 输出在 1024×1024 下非逐字节一致（corr 0.69-0.92）：同基底但
  两种量化 kernel 数值路径不同，8 步采样放大；720×1280 下 qwen-edit 曾逐字节一致。
- 单跑模型（t4_booguedit 等）无 wa4 对照，仅记录基线。

## 5. 建议
- 若想消除"每次重 prewarm"：QW 系 detach 时保留 GPU 权重（速度优先）或接受
  CPU 往返（显存优先），做成可配置。
- 融合加载器 + 内存修复已同步到 01/02 镜像与 WA4 备份。
