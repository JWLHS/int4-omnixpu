"""int4-omnixpu — torchao 依赖安装器（tint4 插件同款机制）。

仅在 tint4 模型需要 torchao 回退（无 omni_xpu_kernel 时）才真正用到。
调用方式：
  - ComfyUI-Manager（插件安装后）
  - python install.py（手动）
  - __init__.py（ComfyUI 启动时自动，仅缺依赖时执行）

注意：omni_xpu_kernel 分 A/B 版且更新频繁，本插件不提供 kernel 自动安装；
kernel 请按 README 说明手动编译/安装。
"""
import sys
import os
import subprocess


MIN_TAO = (0, 17, 0)
PIP_TIMEOUT = 300


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_version(v: str) -> tuple:
    return tuple(int(x) for x in v.split("+")[0].split(".")[:3])


def check_installed() -> str | None:
    try:
        import torchao
        v = _parse_version(torchao.__version__)
        if v >= MIN_TAO:
            return None
        return "old"
    except ImportError:
        return "missing"


def _run(cmd: list, timeout: int = PIP_TIMEOUT) -> subprocess.CompletedProcess:
    print(f"[int4-omnixpu] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _print_output(proc: subprocess.CompletedProcess):
    for line in proc.stdout.split("\n"):
        s = line.strip()
        if s:
            print(f"  {s}")
    if proc.returncode != 0:
        print("[int4-omnixpu] ✗ failed:")
        for line in proc.stderr.strip().split("\n")[-10:]:
            if line.strip():
                print(f"     {line}")


def _install_xpu():
    print()
    print("=" * 58)
    print("  int4-omnixpu — Auto-installing torchao for Intel XPU")
    print("=" * 58)
    print()
    proc = _run([
        sys.executable, "-m", "pip", "install", "--isolated", "torchao",
        "--index-url", "https://download.pytorch.org/whl/xpu",
    ])
    _print_output(proc)
    if proc.returncode != 0:
        print("\n  Manual: pip install torchao --index-url https://download.pytorch.org/whl/xpu")
        return False
    _fix_mps()
    print("\n  ✓ Done — restart ComfyUI.")
    print("=" * 58)
    print()
    return True


_OTHER_DEVICES = {
    "cuda": {"name": "NVIDIA CUDA", "url": ""},
    "rocm": {"name": "AMD ROCm", "url": "https://download.pytorch.org/whl/rocm6.4"},
    "cpu": {"name": "CPU", "url": "https://download.pytorch.org/whl/cpu"},
}


def _install_generic(device: str):
    info = _OTHER_DEVICES.get(device, _OTHER_DEVICES["cpu"])
    print()
    print("=" * 58)
    print(f"  int4-omnixpu — Auto-installing torchao for {info['name']}")
    print("=" * 58)
    print()
    cmd = [sys.executable, "-m", "pip", "install", "torchao"]
    if info["url"]:
        cmd += ["--index-url", info["url"]]
    proc = _run(cmd)
    _print_output(proc)
    if proc.returncode != 0:
        manual = "pip install torchao"
        if info["url"]:
            manual += f" --index-url {info['url']}"
        print(f"\n  Manual: {manual}")
        return False
    print("\n  ✓ Done — restart ComfyUI.")
    print("=" * 58)
    print()
    return True


def _fix_mps():
    """tint4 同款修复：清理 torchao 的 mps 残留目录（若存在）。"""
    try:
        import torchao
    except ImportError:
        return
    d = os.path.join(os.path.dirname(torchao.__file__), "experimental", "ops", "mps")
    if not os.path.isdir(d):
        return
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    print("[int4-omnixpu] cleaned stale torchao mps ops dir")


def ensure_installed(force: bool = False) -> bool:
    """返回 True 表示依赖就绪（无需重启前重试标记）。"""
    if not force and check_installed() is None:
        return True
    device = detect_device()
    if device == "xpu":
        return _install_xpu()
    return _install_generic(device)


if __name__ == "__main__":
    ensure_installed(force=True)
