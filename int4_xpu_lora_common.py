"""
int4_xpu_lora_common.py — Shared LoRA utilities for WA4.
v2.3: FIX — 删除死规则 ".attn.to_out"→".linear2" 与 ".attn.to_qkv_mlp_proj"→".linear1"：
  索引侧（_normalize_index_path）无 linear1/linear2，LoRA 侧这两条永远产生 unmatched；
  且 linear2 先于 ".to_out.0"→".wo" 执行，把 QW 的 attn.to_out.0 打成 linear2.0（60块全 unmatched）
v2.2: FIX — lora_unet_ 全下划线格式支持（保护复合名后转点）
v2.1: _wa4_reset_all_loras handles bake-state rollback.
"""
import logging, re
import torch
from .int4_xpu_loader import _is_quant_linear

log = logging.getLogger("int4-LoRA-Common")


def _get_accelerator_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_layer_path(path: str) -> str | None:
    stripped_prefix = None
    for pf in ["lora_transformer_", "lora_unet_", "lycoris_"]:
        if path.startswith(pf): path = path[len(pf):]; stripped_prefix = pf; break
    if stripped_prefix is None:
        if path.startswith("transformer."): path = path[len("transformer."):]
        elif path.startswith("model.diffusion_model."): path = path[len("model.diffusion_model."):]
        elif path.startswith("diffusion_model."): path = path[len("diffusion_model."):]
    if path.startswith("img_in") or path.startswith("final_layer"): return None
    if path.startswith("text_fusion.layerwise_blocks."):
        path = "blocks." + path[len("text_fusion.layerwise_blocks."):]
    for old, new in [
        ("layers.", "blocks."), ("joint_blocks.", "blocks."),
        ("transformer_blocks.", "blocks."), ("double_blocks.", "blocks."),
        ("single_blocks.", "blocks."), ("single_transformer_blocks.", "blocks."),
        ("transformer_blocks_", "blocks."),
        ("double_blocks_", "blocks."), ("single_blocks_", "blocks."),
        ("single_transformer_blocks_", "blocks."),
    ]:
        if path.startswith(old): path = new + path[len(old):]; break
    if stripped_prefix is not None:
        # v2.2：下划线格式——先保护复合名，再转分隔符，最后还原
        _GUARD = "@@"
        for c in ("add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
                  "img_mlp", "txt_mlp", "img_mod", "txt_mod",
                  "time_text_embed", "timestep_embedder",
                  "norm_added_k", "norm_added_q", "norm_k", "norm_q", "layer_norm"):
            path = path.replace(c, c.replace("_", _GUARD))
        path = path.replace("to_out_0", "to_out.0")
        path = path.replace("_", ".")
        path = path.replace(_GUARD, "_")
    # ── Replace chain（v2.3：删 linear1/linear2 死规则；其余与索引侧一致）──
    path = path.replace(".ff.", ".mlp.").replace(".feed_forward.", ".mlp.").replace(".ffn.", ".mlp.")
    path = path.replace(".img.attn.", ".attn.").replace(".txt.attn.", ".attn.")
    path = path.replace(".img.mlp.", ".img_mlp.").replace(".txt.mlp.", ".txt_mlp.")
    path = path.replace(".attention.", ".attn.")
    path = path.replace(".to_q", ".wq").replace(".to_k", ".wk")
    path = path.replace(".to_v", ".wv").replace(".to_out.0", ".wo")
    path = path.replace(".to_out", ".wo").replace(".to_gate", ".gate")
    path = path.replace(".q_proj", ".wq").replace(".k_proj", ".wk")
    path = path.replace(".v_proj", ".wv").replace(".out_proj", ".wo")
    path = path.replace(".self_attn.q", ".attn.wq")
    path = path.replace(".self_attn.k", ".attn.wk")
    path = path.replace(".self_attn.v", ".attn.wv")
    path = path.replace(".self_attn.o", ".attn.wo")
    path = path.replace(".attn.out", ".attn.wo")
    path = path.replace(".attn.proj", ".attn.wo")
    path = path.replace(".attn.o_proj", ".attn.wo")
    path = path.replace(".gate_proj", ".gate")
    path = path.replace(".up_proj", ".w1")
    path = path.replace(".down_proj", ".w2")
    path = path.replace(".fc1", ".w1").replace(".fc2", ".w2").replace(".fc3", ".w3")
    path = path.replace(".to.q", ".wq").replace(".to.k", ".wk").replace(".to.v", ".wv")
    path = path.replace(".to.out.0", ".wo").replace(".to.out", ".wo")
    return path


_DOUBLE_STREAM_ALIASES = {
    ".attn.": [".img_attn.", ".txt_attn."],
    ".attn1.": [".img_attn1.", ".txt_attn1."],
    ".attn2.": [".img_attn2.", ".txt_attn2."],
    ".mlp.":  [".img_mlp.", ".txt_mlp."],
}

_norm_idx_cache = {}


def _resolve_with_alias(index: dict, norm: str) -> list[object]:
    m = index.get(norm)
    if m is not None: return [m]
    iid = id(index)
    if iid not in _norm_idx_cache:
        _norm_idx_cache[iid] = {k.replace("_", "."): v for k, v in index.items()}
    ni = _norm_idx_cache[iid]
    m = ni.get(norm.replace("_", "."))
    if m is not None: return [m]
    norm_collapsed = re.sub(r'\.attn\d+\.', '.attn.', norm)
    if norm_collapsed != norm:
        m = index.get(norm_collapsed)
        if m is not None: return [m]
        m = ni.get(norm_collapsed.replace("_", "."))
        if m is not None: return [m]
    results = []
    for collapsed, expanded_list in _DOUBLE_STREAM_ALIASES.items():
        if collapsed in norm:
            for exp in expanded_list:
                alias = norm.replace(collapsed, exp)
                m = index.get(alias)
                if m is not None: results.append(m); continue
                m = ni.get(alias.replace("_", "."))
                if m is not None: results.append(m)
    return results


def _auto_detect_format(sd: dict) -> str:
    for key in sd:
        if "single_blocks" in key or "double_blocks" in key: return "bfl"
        if "diffusion_model.blocks" in key or "diffusion_model.layers" in key: return "standard"
    return "unknown"


def _convert_bfl_to_standard(sd: dict) -> dict:
    out = {}
    for key, tensor in sd.items():
        if "qkv.lora" in key or "proj.lora" in key or "ff.lora" in key:
            for prefix in ["double_blocks", "single_blocks"]:
                if key.startswith(prefix): break
            else: out[key] = tensor; continue
            rest = key[len(prefix) + 1:]
            parts = rest.split(".")
            block_num = parts[0]
            attn_type = parts[1] if len(parts) > 1 and "attn" in parts[1] else "attn"
            if "lora_B" in key: lora_type = "up"
            elif "lora_A" in key: lora_type = "down"
            elif "lora_up" in key: lora_type = "up"
            elif "lora_down" in key: lora_type = "down"
            else: out[key] = tensor; continue
            stem = "qkv" if "qkv" in key else "proj"
            std_key = f"diffusion_model.blocks.{block_num}.{attn_type}.{stem}"
            out[f"{std_key}.lora_{lora_type}.weight"] = tensor
        else: out[key] = tensor
    return out


def _rot_quarot_tensor(tensor, H, group_size):
    if H is None or group_size <= 0: return tensor
    if tensor.shape[1] % group_size != 0: return tensor
    Hd = H.to(tensor.device, dtype=torch.float16)
    ng = tensor.shape[1] // group_size
    return (tensor.reshape(tensor.shape[0], ng, group_size) @ Hd.T).reshape(tensor.shape[0], tensor.shape[1])


def _parse_raw_lora_sd(lora_sd: dict) -> dict[str, dict]:
    lora_data: dict[str, dict] = {}
    for key, tensor in lora_sd.items():
        if "lokr_w1" in key:
            idx = key.index("lokr_w1"); lp = _normalize_layer_path(key[:idx].rstrip("."))
            if lp: lora_data.setdefault(lp, {})["lokr_w1"] = tensor; lora_data[lp]["type"] = "lokr"
            continue
        if "lokr_w2" in key:
            idx = key.index("lokr_w2"); lp = _normalize_layer_path(key[:idx].rstrip("."))
            if lp: lora_data.setdefault(lp, {})["lokr_w2"] = tensor; lora_data[lp]["type"] = "lokr"
            continue
        if "lora_up" in key or "lora_B" in key:
            idx = key.index("lora_up") if "lora_up" in key else key.index("lora_B")
            lp = _normalize_layer_path(key[:idx].rstrip("."))
            if lp: lora_data.setdefault(lp, {})["up"] = tensor; lora_data.setdefault(lp, {})["type"] = "standard"
            continue
        if "lora_down" in key or "lora_A" in key:
            idx = key.index("lora_down") if "lora_down" in key else key.index("lora_A")
            lp = _normalize_layer_path(key[:idx].rstrip("."))
            if lp: lora_data.setdefault(lp, {})["down"] = tensor; lora_data.setdefault(lp, {})["type"] = "standard"
            continue
        if key.endswith(".alpha"):
            lp = _normalize_layer_path(key[:-6])
            if lp:
                t = tensor
                lora_data.setdefault(lp, {})["alpha"] = float(t.mean()) if t.numel() > 1 else t.item()
            continue
    return lora_data


def _wa4_reset_all_loras(model) -> None:
    """Full reset: INT4XPULinear entries + bake-state rollback."""
    bm = model.model
    while hasattr(bm, '_orig_mod'): bm = bm._orig_mod
    cpu = torch.device("cpu")
    for m in bm.modules():
        if _is_quant_linear(m):
            object.__setattr__(m, '_wa4_lora_entries', None)
        bs = getattr(m, '_wa4_bake_state', None)
        if bs is None: continue
        hh = bs.pop('_hook_handle', None)
        if hh is not None:
            try: hh.remove()
            except Exception: pass
        applied = bs.pop('_applied', None)
        if applied is not None and hasattr(m, 'weight') and m.weight is not None:
            for delta_cpu, sl, se in applied:
                try:
                    neg = (-delta_cpu).to(device=m.weight.device, dtype=m.weight.dtype)
                    if sl is not None and se is not None:
                        m.weight.data[sl:se].add_(neg)
                    else:
                        m.weight.data.add_(neg)
                except Exception: pass
        bs.pop('_pending', None)
        bs.clear()
    object.__setattr__(model.model, '_wa4_loras', [])
    log.info("[WA4 LoRA] All LoRA entries cleared")

