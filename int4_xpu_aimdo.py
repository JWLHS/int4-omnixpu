"""
int4_xpu_aimdo.py — WA4 × AIMDO XPU 管控 v1.3

v1.3:
  - lora_policy() 恒返回 'normal'：LoRA 与模型一体，始终注入
    （LoRA 的 A/B/delta/写回全部经 AIMDO allocator 分配，纳入 AIMDO 管理，模型管控状态不变）
v1.2:
  - 新增 aimdo_active() 统一查询（loader / LoRA 共用）
  - lora_policy() 曾为 AIMDO 跳过策略——已废弃（实测环境已兼容）
  - devctxs 读取 getattr 加固
v1.1:
  - 检测 AIMDO（只认 XPU 实现 aimdo_xpu；cuda/rocm 无效依赖不理会）
  - getattr 安全：AIMDO 未 init（implementation 不存在）→ 视为 none → wa4 约束生效
  - AIMDO active → wa4 跳过约束（让路，避免 urEventWait 冲突）
"""
import logging
import torch

log = logging.getLogger("int4-AIMDO")

_orig_cuda_sync = getattr(torch.cuda, "synchronize", None)
_noop_sync = lambda *a, **k: None


def aimdo_state() -> str:
    """'none'=未加载XPU实现 / 'active'=aimdo_xpu已接管allocator / 'broken'=dll加载但设备未初始化"""
    try:
        import comfy_aimdo.control as ctrl
    except ImportError:
        return "none"
    if ctrl.lib is None or getattr(ctrl, "implementation", None) != "xpu":
        return "none"
    if not getattr(ctrl, "devctxs", None):
        return "broken"
    return "active"


def aimdo_active() -> bool:
    """统一查询：AIMDO 是否活跃"""
    return aimdo_state() == "active"


def lora_policy() -> str:
    """LoRA 与模型一体：始终正常注入（AIMDO 管控显存，LoRA 走 AIMDO 分配）"""
    return "normal"


def patch_aimdo_xpu(model) -> bool:
    """AIMDO active 时做 XPU 适配，返回是否生效"""
    st = aimdo_state()
    if st == "none":
        return False
    if st == "broken":
        log.warning("[int4] AIMDO: dll 加载但设备未初始化 — wa4 按 none 处理，约束生效")
        return False
    if _orig_cuda_sync is not None:
        torch.cuda.synchronize = _noop_sync
    log.info("[int4] AIMDO: active (XPU allocator replaced) — 模型与 LoRA 均由其管控")
    return True
