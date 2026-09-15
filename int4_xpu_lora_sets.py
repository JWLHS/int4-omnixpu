"""int4_xpu_lora_sets.py — LoRA 规格层（向 ComfyUI 原生 LoRA 节点语义靠拢）

核心思想：**LoRA 集跟着 ModelPatcher 走，不再就地改共享模型**。

  - 节点执行：`model = model.clone()`，生成 token，把"本组规格"登记到 `_SETS`，
    并把 token 写进这一路 patcher 的 `model_options["transformer_options"][SPEC_KEY]`。
    clone 是深拷贝 model_options（comfy/model_patcher.py 的 clone），所以**每条分支
    一份规格**；chained clone 继承上游规格 → 串联＝叠加，并联＝各自独立（原生语义）。
  - 采样时：采样器把 `transformer_options` 传给 `apply_model`，我们在那儿读出 token
    → 拿到这一路的规格 → `sync()` 把模型上的 LoRA 状态**对齐**到规格（多退少补）。
    签名相同就什么都不做（零开销，替代旧的"去重跳过"）。
  - 卸载：`mark_dirty()`（模型上的 LoRA 状态失效）→ 下次采样按**活跃规格**重建。

日志只报结果：每次采样阶段打印"这一刻真正生效的 LoRA"；过程信息一律 debug。
"""
import logging
import os
import time
import weakref

log = logging.getLogger("int4-LoRA")

# transformer_options 里的键。值必须是**字符串 token**，不能是 list：
# comfy/patcher_extension.merge_nested_dicts 对 list 做 extend，会被反复叠加。
SPEC_KEY = "wa4_lora"

_SETS = {}          # token -> {"specs": [...], "ref": weakref(patcher)}
_SEQ = 0
_PARSE_CACHE = {}   # (path, mtime, size) -> 已解析的 LoRA（注入层格式）
_ACTIVE = {"patcher": None}   # 当前这次采样用的是哪一路 patcher（由 prepare_sampling 记录）


def set_active_patcher(patcher):
    _ACTIVE["patcher"] = patcher


def active_patcher():
    return _ACTIVE["patcher"]


# ── 规格表 ────────────────────────────────────────────────────────────

def specs_of(patcher):
    """读某个 ModelPatcher 这一路的 LoRA 规格（没有则空表）。"""
    try:
        opts = patcher.model_options.get("transformer_options") or {}
        token = opts.get(SPEC_KEY)
    except Exception:
        return []
    if not isinstance(token, str):
        return []
    ent = _SETS.get(token)
    return list(ent["specs"]) if ent else []


def specs_of_options(transformer_options):
    """读采样时传下来的 transformer_options 里的规格；没有该键返回 None。"""
    if not isinstance(transformer_options, dict):
        return None
    token = transformer_options.get(SPEC_KEY)
    if token is None:
        return None
    if not isinstance(token, str):
        return []
    ent = _SETS.get(token)
    return list(ent["specs"]) if ent else []


def register(parent_patcher, child_patcher, new_specs):
    """登记"本节点这一路"的规格（= 上游规格 + 本节点新增），返回 token。

    new_specs: [{"name":…, "strength":…, "kind": "loader"/"stack"}]
    """
    global _SEQ
    specs = specs_of(parent_patcher) + [dict(s) for s in new_specs]
    _SEQ += 1
    token = f"wa4-{_SEQ}"
    # 顺手清掉已 GC 的登记，避免表无限增长
    for k in [k for k, v in _SETS.items() if v["ref"]() is None]:
        _SETS.pop(k, None)
    try:
        ref = weakref.ref(child_patcher)
    except TypeError:
        ref = lambda: child_patcher
    _SETS[token] = {"specs": specs, "ref": ref}
    try:
        opts = child_patcher.model_options.setdefault("transformer_options", {})
        opts[SPEC_KEY] = token
    except Exception as e:
        log.debug("[int4 LoRA] 写入规格失败：%s", e)
    return token


# ── 解析缓存（替代旧的 0 层负缓存）────────────────────────────────────

def parse_lora(path):
    """按 (路径, mtime, size) 缓存已解析的 LoRA，避免重复读文件。"""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except Exception:
        key = (path, 0, 0)
    hit = _PARSE_CACHE.get(key)
    if hit is not None:
        return hit
    from .int4_xpu_lora_loader import parse_lora_file
    data = parse_lora_file(path)
    if len(_PARSE_CACHE) > 32:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[key] = data
    return data


# ── 已应用状态（挂在 base model 上）──────────────────────────────────

def applied_state(model):
    st = getattr(model.model, "_wa4_applied_loras", None)
    if st is None:
        st = {"sigs": (), "detail": {}, "dirty": True, "last_t": None}
        object.__setattr__(model.model, "_wa4_applied_loras", st)
    return st


def mark_dirty(model):
    """模型卸载：LoRA 状态失效（权重被释放/还原）→ 下次采样按活跃规格重建。"""
    st = applied_state(model)
    st["sigs"] = ()
    st["detail"] = {}
    st["dirty"] = True


def _sig(spec, index):
    """(唯一键, 强度)。键带上序号：同一个 LoRA 在链里出现两次时，两份 delta 独立记账
    （原生语义：同一个 LoRA 叠两次 = 双倍强度），互不覆盖。"""
    key = spec.get("key") or f"{spec.get('name')}::{index}"
    return (key, round(float(spec.get("strength", 1.0)), 6))


def _is_new_stage(st, timestep):
    """timestep/sigma 回升 = 新一轮降噪循环（新的一次采样）→ 要再报一次日志。"""
    if timestep is None:
        return False
    try:
        t = float(timestep) if not hasattr(timestep, "numel") else float(timestep.max())
    except Exception:
        return False
    last = st.get("last_t")
    st["last_t"] = t
    return last is None or t > last


# ── 对齐（多退少补）+ 阶段日志 ───────────────────────────────────────

def sync(model, specs, timestep=None):
    """把模型上的 LoRA 状态对齐到 specs。返回本次实际注入的条数。"""
    from .int4_xpu_lora_loader import apply_lora, remove_lora

    st = applied_state(model)
    want = [_sig(s, i) for i, s in enumerate(specs)]
    changed = st["dirty"] or tuple(want) != tuple(st["sigs"])
    new_stage = _is_new_stage(st, timestep)
    if not changed and not new_stage:
        return 0

    injected, removed, injected_detail = [], 0, {}
    if changed:
        have = {n: s for n, s in st["sigs"]}
        for name in list(have):
            if name not in {n for n, _s in want}:
                remove_lora(model, name)
                removed += 1
        for index, spec in enumerate(specs):
            name, strength = _sig(spec, index)
            if not st["dirty"] and have.get(name) == strength and name in st["detail"]:
                continue
            t0 = time.perf_counter()
            stats = apply_lora(model, spec.get("name"), float(strength), entry_key=name)
            injected_detail[name] = {
                "quant": stats.get("quant", 0), "bake": stats.get("bake", 0),
                "unmatched": stats.get("unmatched", 0),
                "elapsed": time.perf_counter() - t0,
            }
            injected.append(name)
        st["sigs"] = tuple(want)
        keep_keys = {k for k, _s in want}
        st["detail"] = {**{k: v for k, v in st["detail"].items() if k in keep_keys},
                        **injected_detail}
        st["dirty"] = False
        if injected:
            # 重建过 LoRA 后重新武装权重预热（与加载路径一致；已在显存上的层是空操作）
            try:
                from .int4_xpu_loader import _wa4_arm_prewarm
                _wa4_arm_prewarm(model)
            except Exception as e:
                log.debug("[int4 LoRA] 预热重武装失败：%s", e)

    _stage_log(specs, st, injected, new_stage, changed)
    return len(injected)


def _stage_log(specs, st, injected, new_stage, changed):
    """每次采样阶段打印一组"本次使用了哪些 LoRA"；过程信息全部 debug。"""
    if not specs:
        if new_stage and changed:
            log.info("[int4 LoRA] ✓ 本次采样：无 LoRA")
        return
    total_q = total_b = 0
    any_reuse = False
    kinds = set()
    for index, spec in enumerate(specs):
        name = spec.get("name")
        strength = float(spec.get("strength", 1.0))
        key = _sig(spec, index)[0]
        d = st["detail"].get(key) or {}
        q, b = int(d.get("quant") or 0), int(d.get("bake") or 0)
        total_q += q
        total_b += b
        kind = spec.get("kind") or "loader"
        kinds.add(kind)
        prefix = "[int4 Stack]" if kind == "stack" else "[int4 LoRA]"
        parts = _parts(q, b)
        if key in injected:
            log.info("%s ✓ 注入 %s | %s | strength=%s | %.2fs%s",
                     prefix, name, parts, strength,
                     float((d.get("elapsed") or 0.0)), "")
        else:
            any_reuse = True
            log.info("%s ✓ 注入 %s | %s | strength=%s | 复用",
                     prefix, name, parts, strength)
    summary_prefix = "[int4 Stack]" if "stack" in kinds else "[int4 LoRA]"
    if any_reuse and not injected:
        log.info("%s ✓ 共 %d 个 LoRA（%d 量化层 + %d bake 层）| 复用",
                 summary_prefix, len(specs), total_q, total_b)
    else:
        log.info("%s ✓ 共 %d 个 LoRA（%d 量化层 + %d bake 层）",
                 summary_prefix, len(specs), total_q, total_b)


def _parts(q, b):
    if not q and not b:
        return "0 layers"
    out = []
    if q: out.append(f"{q} quant")
    if b: out.append(f"{b} bake")
    return " + ".join(out)
