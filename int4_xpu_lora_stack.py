"""int4_xpu_lora_stack.py — INT4XPU LoRA Stack（薄壳节点）

与 INT4XPULoRALoader 同一套规格机制：只登记"这一路要哪些 LoRA"，
采样时由 int4_xpu_lora_sets 按活跃规格对齐（语义与原生 LoRA 节点一致）。
最多 8 个；同一个 LoRA 填多次 = 多份（原生叠加语义）。
"""
import logging

import folder_paths

from . import int4_xpu_lora_sets as int4_lora_sets
from .int4_xpu_lora_loader import install_detach

log = logging.getLogger("int4-LoRA-Stack")


class INT4XPULoRAStack:
    NAME = "INT4XPU LoRA Stack"
    CATEGORY = "int4"

    @classmethod
    def INPUT_TYPES(cls):
        inp = {"required": {"model": ("MODEL", {"tooltip": "From int4XPUModelLoader"})},
               "optional": {}}
        for i in range(1, 9):
            inp["optional"][f"lora_name_{i}"] = (["None"] + folder_paths.get_filename_list("loras"),)
            inp["optional"][f"strength_{i}"] = (
                "FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        return inp

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    DESCRIPTION = "把最多 8 个 LoRA 作为一个整体叠加到模型上（采样时应用）。"

    def apply(self, model, **kwargs):
        parent = model
        model = model.clone()
        install_detach(model)
        specs = []
        for i in range(1, 9):
            name = kwargs.get(f"lora_name_{i}")
            strength = kwargs.get(f"strength_{i}", 1.0)
            if name is None or name == "None" or name == "":
                continue
            if abs(float(strength)) < 1e-5:
                # 0 强度也登记规格：可撤掉链里前面同名的 LoRA（归零=撤销）
                specs.append({"name": name, "strength": 0.0, "kind": "stack"})
                continue
            if folder_paths.get_full_path("loras", name) is None:
                log.warning("[int4 Stack] LoRA 不存在：%s", name)
                continue
            specs.append({"name": name, "strength": float(strength), "kind": "stack"})
        if specs:
            int4_lora_sets.register(parent, model, specs)
        return (model,)


NODE_CLASS_MAPPINGS = {"INT4XPULoRAStack": INT4XPULoRAStack}
NODE_DISPLAY_NAME_MAPPINGS = {"INT4XPULoRAStack": "INT4XPU LoRA Stack (up to 8)"}
