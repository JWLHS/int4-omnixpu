"""int4-omnixpu — torchao 依赖安装器（ComfyUI-Manager 入口）。

实现全部在 int4_xpu_install.py（__init__.py 启动时也走同一实现），
这里只是薄包装，避免双份代码维护。
"""
from int4_xpu_install import ensure_installed


if __name__ == "__main__":
    ensure_installed(force=True)
