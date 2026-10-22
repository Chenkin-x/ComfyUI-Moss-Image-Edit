"""Load MOSS editing weights and edit with one to four reference images."""
from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import torch
from PIL import Image

import comfy.model_management as model_management
from comfy.utils import ProgressBar
import folder_paths

from .moss_runtime.weights import read_adapter_options
from .moss_runtime.components import encode_prompts, load_moss_module, load_transformer, torch_dtype
from .moss_runtime.weights import inject_lora, load_adapter
from .moss_runtime.model import MossEditModel
from .moss_runtime.processing import decode_latents, encode_image, image_to_tensor, load_encoding_components
from .moss_runtime.sampling import sample_latents


def _model_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(folder_paths.models_dir) / path).resolve(strict=True)


def _empty_cache():
    gc.collect()
    model_management.soft_empty_cache()


def _reference_tensor(image: torch.Tensor, max_edge: int) -> torch.Tensor:
    """Comfy IMAGE -> the same Lanczos resize and /16 rounding as CLI inference."""
    if image.shape[0] != 1:
        raise ValueError("Connect one image to each reference input; image batches are not reference lists.")
    pixels = image[0, :, :, :3].detach().float().cpu().clamp(0, 1)
    picture = Image.fromarray((pixels * 255).round().to(torch.uint8).numpy())
    width, height = picture.size
    limit = max_edge // 16 * 16
    scale = min(1.0, limit / max(width, height))

    def aligned(value):
        return min(limit, max(16, int(math.floor(value * scale / 16 + 0.5)) * 16))

    picture = picture.resize((aligned(width), aligned(height)), Image.Resampling.LANCZOS)
    return image_to_tensor(picture)


class _MossPipeline:
    """Keep loaded weights on CPU between requests; use one GPU stage at a time."""

    def __init__(self, snapshot: Path, adapter: Path):
        self.snapshot = snapshot
        self.adapter = adapter
        self.options, self.metadata = read_adapter_options(adapter)
        self.encoding = None
        self.editor = None

    def load_encoders(self):
        if self.encoding is None:
            logging.info("MOSS Image Edit: loading text encoder and VAE on CPU")
            encoding = load_encoding_components(
                self.snapshot, device="cpu", dtype=torch.bfloat16, vae_dtype=torch.float32,
            )
            from .moss_runtime.quantization import quantize_frozen_linears
            quantize_frozen_linears(encoding.prompt_components.text_encoder)
            encoding.prompt_components.offload_text_encoder = True
            self.encoding = encoding
        return self.encoding

    def load_editor(self):
        if self.editor is None:
            logging.info("MOSS Image Edit: loading DiT and editing weights on CPU")
            model = load_transformer(self.snapshot, device="cpu", dtype=torch.bfloat16)
            options = {**self.options, "param_dtype": torch_dtype(self.options["param_dtype"])}
            inject_lora(model, **options)
            load_adapter(model, self.adapter)
            model.requires_grad_(False).eval()
            from .moss_runtime.quantization import quantize_frozen_linears
            quantize_frozen_linears(model)
            self.editor = MossEditModel(model).eval()
        return self.editor

    def offload(self):
        if self.encoding is not None:
            self.encoding.prompt_components.text_encoder.to("cpu")
            self.encoding.vae.to("cpu")
        if self.editor is not None:
            self.editor.to("cpu")
        _empty_cache()


class MossImageEditLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "snapshot_dir": ("STRING", {"default": "moss_image/MOSS-Image-SFT"}),
            "adapter_dir": ("STRING", {"default": "moss_image/Moss-Image-Edit"}),
        }}

    RETURN_TYPES = ("MOSS_EDIT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MOSS Image Edit"
    DESCRIPTION = "Select the full MOSS-Image-SFT directory and the editing adapter directory. Relative paths start at ComfyUI/models."

    def load(self, snapshot_dir, adapter_dir):
        # Actual weights load on first edit, then stay cached on CPU.
        return (_MossPipeline(_model_path(snapshot_dir), _model_path(adapter_dir)),)


class MossImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOSS_EDIT_MODEL",),
                "reference_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1, "control_after_generate": True}),
                "reference_max_edge": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "inference_shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "reference_2": ("IMAGE",),
                "reference_3": ("IMAGE",),
                "reference_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "MOSS Image Edit"
    DESCRIPTION = "Provide 1–4 reference images in order, describe the changes, and generate an edited image."

    # ComfyUI normally runs nodes in inference_mode. The shared reference-cache
    # validation uses tensor version counters, so create all model/cache tensors
    # in no_grad mode with inference_mode explicitly disabled instead.
    @torch.inference_mode(False)
    @torch.no_grad()
    def edit(self, model, reference_1, prompt, negative_prompt, width, height, steps,
             cfg, seed, reference_max_edge, inference_shift,
             reference_2=None, reference_3=None, reference_4=None):
        references = [image for image in (reference_1, reference_2, reference_3, reference_4) if image is not None]
        device = model_management.get_torch_device()
        model_management.throw_exception_if_processing_interrupted()
        model_management.unload_all_models()
        _empty_cache()
        try:
            encoding = model.load_encoders()
            encoding.prompt_components.text_encoder_device = device
            logging.info("MOSS Image Edit: encoding instruction")
            conditions = encode_prompts(
                encoding.prompt_components, [prompt],
                negative_prompts=[negative_prompt] if cfg != 1 else None,
                max_sequence_length=4096, device="cpu",
            )
            _empty_cache()
            model_management.throw_exception_if_processing_interrupted()
            logging.info("MOSS Image Edit: encoding %d reference image(s)", len(references))
            encoding.vae.to(device)
            reference_latents = [
                encode_image(encoding.vae, _reference_tensor(image, reference_max_edge), sample_posterior=False).cpu()
                for image in references
            ]
            encoding.vae.to("cpu")
            _empty_cache()
            model_management.throw_exception_if_processing_interrupted()

            editor = model.load_editor().to(device)
            scheduler = load_moss_module(model.snapshot).MOSSImageScheduler.from_pretrained(
                str(model.snapshot / "scheduler"), local_files_only=True,
                shift=inference_shift, training_shift=1.0,
            )
            progress = ProgressBar(steps)

            def update(index, total):
                model_management.throw_exception_if_processing_interrupted()
                progress.update_absolute(index, total)

            latents = sample_latents(
                editor, scheduler, reference_latents, conditions.positive, conditions.negative,
                height=height, width=width, steps=steps, cfg=cfg, seed=seed, device=device,
                latent_channels=int(editor.config.latent_channels), use_kv_cache=True, progress=update,
            )
            editor.to("cpu")
            _empty_cache()
            model_management.throw_exception_if_processing_interrupted()
            logging.info("MOSS Image Edit: decoding result")
            decoded = decode_latents(encoding.vae.to(device), latents)
            image = (decoded.float().cpu() + 1).mul_(0.5).clamp_(0, 1)
            return (image.permute(0, 2, 3, 1).contiguous(),)
        finally:
            # Also release device memory when a user cancels sampling or a stage
            # fails. ComfyUI's node cache retains only the CPU-side model handle.
            model.offload()


NODE_CLASS_MAPPINGS = {
    "MossImageEditLoader": MossImageEditLoader,
    "MossImageEdit": MossImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossImageEditLoader": "Load MOSS Image Edit",
    "MossImageEdit": "MOSS Image Edit",
}
