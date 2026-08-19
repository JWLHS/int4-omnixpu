"""
INT4XPU Model Quantizer v1.0
— FP16/BF16/FP32/FP8/INT8 → INT4 per-group absmax quantization.
Auto-strips embedded VAE/CLIP, cleans residual FP8.

v1.0（重建版）:
  weight_dtype 选项（fp16 / bf16）— 手动选择，写入 __w4a4_weight_dtype__ 标记
  loader 读标记自动跟随，不再靠 cfg 猜
  scale 保持 f16（oneDNN 原版算子要求）
"""
import os
import torch
import logging
import folder_paths
import comfy.utils
from safetensors.torch import save_file

log = logging.getLogger("wa4")

INT4_MAX = 8

_FP8_TYPES = set()
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
	_t = getattr(torch, _name, None)
	if _t is not None:
		_FP8_TYPES.add(_t)

_STRIP_TEXT_VAE_SUBSTRINGS = ("text_encoder", "clip", "vae", "cond_stage")

def _is_text_vae_key(key: str) -> bool:
	"""True = text encoder / vision / VAE / condition 模块键，不参与 W4A16 量化。"""
	return any(s in key for s in _STRIP_TEXT_VAE_SUBSTRINGS)

_EXCLUSIONS = {
	"flux2": [
		"img_in", "time_in", "guidance_in", "txt_in", "final_layer",
		"double_stream_modulation", "single_stream_modulation",
	],
	"z-image": [
		"cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
		"context_refiner", "final_layer", "noise_refiner", "adaLN",
		"x_pad_token", "cap_embedder.0",
		"attention_norm1", "attention_norm2",
		"ffn_norm1", "ffn_norm2", "k_norm", "q_norm",
		"feed_forward.w2",
		"feed_forward.w1", "feed_forward.w3",
	],
	"chroma": [
		"distilled_guidance_layer", "final_layer", "img_in", "txt_in",
		"nerf_image_embedder", "nerf_blocks",
		"nerf_final_layer_conv", "__x0__",
	],
	"wan": [
		"patch_embedding", "text_embedding", "time_embedding",
		"time_projection", "head", "img_emb", "motion_encoder",
		"modulation", "norm_q", "norm_k", "norm3",
	],
	"ltx2": [
		"adaln_single", "audio_adaln_single",
		"audio_caption_projection", "audio_patchify_proj",
		"audio_proj_out", "audio_scale_shift_table",
		"av_ca_a2v_gate_adaln_single",
		"av_ca_audio_scale_shift_adaln_single",
		"av_ca_v2a_gate_adaln_single",
		"av_ca_video_scale_shift_adaln_single",
		"caption_projection", "patchify_proj", "proj_out",
		"scale_shift_table", "learnable_registers",
		"q_norm", "k_norm",
	],
	"qwen": [
		"text_encoders", "time_text_embed", "img_in",
		"norm_out", "proj_out", "txt_in",
		"norm_added_k", "norm_added_q", "norm_k", "norm_q",
		"txt_norm", "transformer_blocks.0.img_mod.1",
	],
	"ernie": [
		"time", "x_embedder", "adaLN", "final",
		"text_proj", "norm", "layers.0.", "layers.35",
	],
	"hidream": [
		"patch_embedding", "time_text_embed", "norm_out", "proj_out",
	],
	"boogu": [
		"embed", "refine", "norm_out",
	],
	"krea2": [
		"first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
	],
	"ideogram4": [
		"embed_image_indicator", "t_embedding", "proj",
	],
	"anima": [
		"adaln", "x_embedder", "final_layer", "t_embedder",
		"llm_adapter", "cross_attn",
	],
	"sd3": [
		"x_embedder", "y_embedder", "context_embedder",
		"final_layer", "pos_embed",
	],
	"flux": [
		"img_in", "txt_in", "time_in", "vector_in",
		"guidance_in", "final_layer",
		"img_mod.lin", "txt_mod.lin", "modulation.lin",
	],
	"hunyuan_video": [
		"img_in", "txt_in", "time_in", "final_layer",
		"vector_in", "guidance_in", "vision_in", "byt5_in",
	],
	"hunyuan3d": [
		"latent_in", "cond_in", "final_layer", "guidance_in",
	],
	"auraflow": [
		"positional_encoding", "cond_seq_linear",
	],
	"hydit": [
		"x_embedder", "extra_embedder", "final_layer",
		"time_embed", "mlp_t5",
	],
	"mochi": [
		"t5_yproj", "t5_yembed", "x_embed", "final_layer",
	],
	"pixart": [
		"t_block", "pos_embed", "y_embedder",
		"ar_embedder", "x_embedder", "final_layer",
	],
	"cosmos": [
		"x_embedder", "final_layer", "adaln", "t_embedder",
	],
	"cogvideox": [
		"patch_embed", "proj_out", "ofs_embedding",
	],
	"lumina2": [
		"cap_embedder", "noise_refiner", "x_embedder",
		"final_layer", "t_embedder",
	],
	"omnigen2": [
		"time_caption_embed", "x_embedder", "final_layer",
	],
	"lens": [
		"img_in", "proj_out", "txt_norm",
	],
	"kandinsky5": [
		"visual_embeddings", "time_embeddings",
	],
	"hidream_o1": [
		"t_embedder1", "x_embedder.proj1", "final_layer",
	],
	"seedvr2": [
		"x_embedder", "final_layer",
	],
	"h3": [
		"adaln_proj", "video_patch_proj", "audio_patch_proj",
		"condition_proj", "final_layer", "time_embedder", "time_embed",
	],
	"auto": [],
}

MODEL_TYPES = [
	"flux2 (Flux.2)", "flux (Flux.1 dev/schnell)", "sd3 (SD3 / SD3.5)",
	"hunyuan_video (Hunyuan Video)", "hunyuan3d (Hunyuan3D 2.x)",
	"pixart (PixArt Alpha / Sigma)", "hydit (Hunyuan DiT)",
	"auraflow (AuraFlow)", "mochi (Mochi Preview)", "cosmos (Cosmos)",
	"cogvideox (CogVideoX)", "lumina2 (Lumina 2 / NewBie)",
	"omnigen2 (OmniGen 2)", "lens (Lens)", "kandinsky5 (Kandinsky 5)",
	"hidream_o1 (HiDream-O1)", "seedvr2 (SeedVR 2)", "z-image (Z-Image)",
	"chroma (Chroma / Radiance)", "wan (Wan 2.1)", "ltx2 (LTX Video 2)",
	"qwen (Qwen Image)", "ernie (Ernie Image)", "hidream (HiDream Full)",
	"boogu (Boogu)", "krea2 (Krea 2)", "ideogram4 (Ideogram 4)",
	"anima (Anima / Cosmos Predict2)", "auto (Auto-detect)",
	"h3 (MiniMax H3)",
]


def model_type_key(d):
	return d.split(" (")[0] if " (" in d else d


def _is_excluded(key, mt):
	for p in _EXCLUSIONS.get(mt, []):
		if p in key:
			return True
	return False


def _should_quantize(key, tensor, mt):
	if tensor.ndim != 2:
		return False
	if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, *list(_FP8_TYPES), torch.int8):
		return False
	if _is_excluded(key, mt):
		return False
	return True


def _get_hadamard(gs, device="cpu"):
	try:
		from .int4_xpu_quarot import build_hadamard
		return build_hadamard(gs, device=device, dtype=torch.float32)
	except ImportError:
		return None


def _rotate_weight_tensor(w, H, gs):
	try:
		from .int4_xpu_quarot import rotate_weight
		return rotate_weight(w, H, gs)
	except ImportError:
		return w


def _pack_int4(w):
	lo = w[..., 0::2].to(torch.int32) & 0x0F
	hi = w[..., 1::2].to(torch.int32) & 0x0F
	return (lo | (hi << 4)).to(torch.int8)


def _quantize_weight(w):
	o, i = w.shape
	for gs in (64, 32):
		if i % gs == 0:
			wf = w.float()
			groups = wf.reshape(o, i // gs, gs)
			absmax = groups.abs().amax(dim=-1).clamp(min=1e-10)
			scales = absmax / INT4_MAX
			# int4 有效范围是 [-8, 7]；scale=absmax/8 时 absmax 元素会 round 到
			# 8，必须饱和到 7（单点饱和，比整体只用 [-7,7] 的动态范围损失小得多）。
			q = (groups / scales.unsqueeze(-1)).round().clamp(-INT4_MAX, INT4_MAX - 1).to(torch.int8)
			return _pack_int4(q.reshape(o, i)), scales.t().contiguous().to(torch.float16), gs
	raise ValueError(f"in_features {i}: no usable group_size")


class int4XPUModelQuantizer:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"unet_name": (folder_paths.get_filename_list("diffusion_models"),),
				"model_type": (MODEL_TYPES, {"default": "flux2 (Flux.2)"}),
				"output_filename": ("STRING", {"default": "model_wa4"}),
				"device": (["cpu", "xpu"], {"default": "xpu"}),
				"enable_quarot": ("BOOLEAN", {"default": False}),
				"quarot_group_size": ("INT", {"default": 128, "min": 64, "max": 256, "step": 64}),
				"weight_dtype": (["fp16", "bf16"], {"default": "fp16"}),
			}
		}

	RETURN_TYPES = ()
	FUNCTION = "quantize"
	CATEGORY = "wa4"
	TITLE = "INT4XPU Model Quantizer v1.0"
	OUTPUT_NODE = True

	def quantize(self, unet_name, model_type, output_filename, device,
				 enable_quarot=False, quarot_group_size=128, weight_dtype="fp16"):
		model_type = model_type_key(model_type)
		unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
		sd = comfy.utils.load_torch_file(unet_path, safe_load=True)

		# ── 剥离嵌入的 VAE / 文本编码器（特征过滤，兼容 AIO 融合键名）──
		stripped = 0
		for key in list(sd.keys()):
			if _is_text_vae_key(key):
				del sd[key]
				stripped += 1
		if stripped:
			log.info("[int4] Stripped %d VAE/text_encoder keys from merged model", stripped)

		dev = torch.device(device)

		H = None
		qa = False
		if enable_quarot:
			H = _get_hadamard(quarot_group_size, device=str(dev))
			if H is not None:
				qa = True

		output_sd = {}
		quant_count = 0

		for key, tensor in sd.items():
			if key.endswith(".weight") and tensor.ndim == 2:
				base = key[:-7]
				if not _should_quantize(key, tensor, model_type):
					output_sd[key] = tensor
					continue
				dtype = tensor.dtype
				wtq = None
				if dtype in (torch.float16, torch.bfloat16, torch.float32):
					wtq = tensor
				elif dtype in _FP8_TYPES:
					wtq = tensor.to(torch.float16)
				elif dtype == torch.int8:
					ws = sd.get(f"{base}.weight_scale")
					if ws is not None:
						wf = tensor.float()
						if ws.ndim >= 1 and ws.shape[0] > 1:
							wf = wf * ws.view(-1, 1)
						else:
							wf = wf * ws
						wtq = wf.to(torch.float16)
				if wtq is not None:
					try:
						w = wtq.to(dev)
						if H is not None and w.shape[1] % quarot_group_size == 0:
							try:
								w = _rotate_weight_tensor(w, H, quarot_group_size)
							except ValueError:
								pass
						packed, scales, gs = _quantize_weight(w)
						output_sd[key] = packed.cpu()
						output_sd[f"{base}.weight_scale"] = scales.cpu()
						output_sd[f"{base}.w4a4_group_size"] = torch.tensor([gs], dtype=torch.int32)
						quant_count += 1
					except ValueError as e:
						log.warning("[int4] Skipped %s: %s", key, e)
						output_sd[key] = tensor
					continue
			if key not in output_sd:
				output_sd[key] = tensor

		# ── 按选项统一转换权重 dtype（fp16/bf16/FP8 → target_dt）──
		target_dt = torch.float16 if weight_dtype == "fp16" else torch.bfloat16
		converted = 0
		for key, tensor in output_sd.items():
			if isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.float16, torch.bfloat16, *_FP8_TYPES):
				if tensor.dtype != target_dt:
					output_sd[key] = tensor.to(target_dt)
					converted += 1
		if converted:
			log.info("[int4] Weights → %s (%d converted)", weight_dtype, converted)

		# ── 写入权重 dtype 标记（loader 读它自动跟随）──
		output_sd["__w4a4_weight_dtype__"] = torch.tensor(1 if target_dt == torch.bfloat16 else 0, dtype=torch.uint8)

		output_sd["__w4a4_quarot__"] = torch.tensor(1 if qa else 0, dtype=torch.uint8)
		if qa:
			output_sd["__w4a4_quarot_group_size__"] = torch.tensor(quarot_group_size, dtype=torch.int32)

		dst = os.path.join(folder_paths.get_output_directory(), f"{output_filename}.safetensors")
		save_file(output_sd, dst)
		log.info("[int4] Saved: %s | %d quantized | QuaRot=%s | dtype=%s",
				 dst, quant_count, "ON" if qa else "OFF", weight_dtype)
		return ()


NODE_CLASS_MAPPINGS = {"int4XPUModelQuantizer": int4XPUModelQuantizer}
NODE_DISPLAY_NAME_MAPPINGS = {"int4XPUModelQuantizer": "INT4XPU Model Quantizer v1.0"}
