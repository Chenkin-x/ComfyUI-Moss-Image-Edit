"""MOSS editing through native ComfyUI models, conditioning and sampling."""
import json
from functools import wraps
from types import MethodType

import torch
from safetensors import safe_open

import comfy.conds
import comfy.lora
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import nodes as comfy_nodes
import node_helpers
from comfy_api.latest import io

from .native_edit import edit_forward

EDIT_CONTRACT = "ref-prefix-only-t0__target-text-original-moss-v1"


class MossEditWeights(torch.nn.Module):
    def __init__(self, state, config):
        super().__init__()
        self.config = config
        expected = {}
        for target, spec in config["modules"].items():
            expected[target + ".lora_A"] = (spec["rank"], spec["in_features"])
            expected[target + ".lora_B"] = (spec["out_features"], spec["rank"])
        if state.keys() != expected.keys():
            raise ValueError("MOSS Edit patch tensors do not match its embedded adapter_config.")
        for key, value in state.items():
            if tuple(value.shape) != expected[key]:
                raise ValueError(f"MOSS Edit patch has an invalid shape for {key}.")
            parts = key.split(".")
            parent = self
            for part in parts[:-1]:
                if part not in parent._modules:
                    parent.add_module(part, torch.nn.Module())
                parent = parent._modules[part]
            parent.register_parameter(parts[-1], torch.nn.Parameter(value))


def register_patch_format():
    # Core currently has no model-patch format registry. Extend only its loader;
    # all other formats retain their original loader and the same node ID.
    if "ModelPatchLoader" not in comfy_nodes.NODE_CLASS_MAPPINGS:
        raise ImportError("Native ModelPatchLoader is not registered; check ComfyUI's built-in node import errors.")
    loader_class = comfy_nodes.NODE_CLASS_MAPPINGS["ModelPatchLoader"]
    original = loader_class.load_model_patch
    if getattr(original, "_moss_edit_format", False):
        return

    @wraps(original)
    def load_model_patch(self, name):
        path = folder_paths.get_full_path_or_raise("model_patches", name)
        if not path.lower().endswith(".safetensors"):
            return original(self, name)
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        config = json.loads(metadata.get("adapter_config", "{}"))
        if config.get("format") != "moss-edit-lora":
            return original(self, name)
        if config.get("format_version") != 1 or config.get("metadata", {}).get("edit_contract") != EDIT_CONTRACT:
            raise ValueError("Unsupported MOSS Edit patch contract; use a version 1 ref-prefix-only-t0 adapter.")
        state = comfy.utils.load_torch_file(path, safe_load=True)
        patch = MossEditWeights(state, config)
        return (comfy.model_patcher.CoreModelPatcher(
            patch, load_device=comfy.model_management.get_torch_device(),
            offload_device=torch.device("cpu")),)

    load_model_patch._moss_edit_format = True
    loader_class.load_model_patch = load_model_patch


class ApplyMossImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ApplyMossImageEdit", display_name="Apply MOSS Edit", category="model/patches",
            description="待编辑图像在前，附加参考图按编号排列；每张图编码一次并同步到正负条件。多图输入实际参与推理，效果取决于 Edit LoRA 训练。SLG 在外部显式连接。 / Source first, then numbered references; all images participate in inference. Multi-image quality depends on the trained adapter.",
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae", tooltip="匹配的 FLUX.2 VAE / Matching FLUX.2 VAE"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Image.Input("image", display_name="待编辑图像 / source image",
                               tooltip="放在参考序列首位的待编辑原图；输出从空 latent 采样。批次中的图片按顺序作为多张参考，不是分别出图。 / First in the reference sequence. A batch supplies multiple references, not independent edits."),
                io.Int.Input("reference_max_edge", default=512, min=16, max=8192, step=16,
                             tooltip="各图分别保持比例，参考长边上限默认512，宽高对齐16。与输出尺寸独立。 / Per-image long-edge limit; preserve aspect ratio, align to 16. Independent of output size."),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.05),
                io.Autogrow.Input("reference_images", display_name="附加参考图像 / additional references", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("reference_image", optional=True), prefix="reference_", min=0, max=100),
                    tooltip="可留空；连线后增加输入口，按 reference_0、1、2… 排列。多图已接入推理，但当前 LoRA 效果仍需评估。100 是当前 ComfyUI 动态接口上限，不是模型能力或推荐图数。 / Optional ordered references; current adapter quality varies. 100 is the framework socket limit, not a model limit."),
            ],
            outputs=[io.Model.Output(display_name="model"), io.Conditioning.Output(display_name="positive"),
                     io.Conditioning.Output(display_name="negative")],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, positive, negative, image, reference_max_edge, strength, reference_images=None):
        if not isinstance(model.model, comfy.model_base.MOSSImage):
            raise ValueError("Apply MOSS Edit needs a MOSS Image model from Load Diffusion Model.")
        if not isinstance(model_patch.model, MossEditWeights):
            raise ValueError("Select a MOSS Edit adapter in Load Model Patch.")
        if strength == 0:
            return io.NodeOutput(model, positive, negative)
        patch = model_patch.model
        keys = comfy.lora.model_lora_keys_unet(model.model, {})
        state = patch.state_dict()
        lora = {}
        for target, spec in patch.config["modules"].items():
            prefix = "diffusion_model." + target
            if prefix not in keys:
                raise ValueError(f"MOSS model does not support edit patch target: {target}")
            lora[prefix + ".lora_A.weight"] = state[target + ".lora_A"]
            lora[prefix + ".lora_B.weight"] = state[target + ".lora_B"]
            lora[prefix + ".alpha"] = torch.tensor(spec["alpha"])
        loaded = comfy.lora.load_lora(lora, keys)
        result = model.clone()
        applied = result.add_patches(loaded, strength)
        if len(applied) != len(loaded) or len(loaded) != len(patch.config["modules"]):
            raise ValueError("Some MOSS Edit weights could not be applied; update ComfyUI's native MOSS support.")
        original = model.get_model_object("extra_conds")
        def extra_conds(_self, **kwargs):
            out = original(**kwargs)
            refs = kwargs.get("reference_latents", [])
            if refs:
                out["reference_latents"] = comfy.conds.CONDList([_self.process_latent_in(ref) for ref in refs])
            return out
        result.add_object_patch("extra_conds", MethodType(extra_conds, model.model))
        original_forward = model.get_model_object("diffusion_model._forward")
        def forward(_self, x, timestep, context, transformer_options=None, **kwargs):
            if not kwargs.get("reference_latents"):
                return original_forward(x, timestep, context, transformer_options or {}, **kwargs)
            return edit_forward(_self, x, timestep, context, transformer_options, **kwargs)
        result.add_object_patch("diffusion_model._forward", MethodType(forward, model.model.diffusion_model))
        ordered_images = [image]
        for name in sorted(reference_images or {}, key=lambda name: int(name.removeprefix("reference_"))):
            if reference_images[name] is not None:
                ordered_images.append(reference_images[name])
        references = []
        for images in ordered_images:
            for source in images.split(1):
                height, width = source.shape[1:3]
                scale = min(1.0, reference_max_edge / max(height, width))
                aligned_height = max(16, round(height * scale / 16) * 16)
                aligned_width = max(16, round(width * scale / 16) * 16)
                if (height, width) != (aligned_height, aligned_width):
                    source = comfy.utils.common_upscale(source.movedim(-1, 1), aligned_width, aligned_height, "lanczos", "disabled").movedim(1, -1)
                reference = vae.encode(source[:, :, :, :3])
                if reference.ndim != 4 or reference.shape[1] != model.model.diffusion_model.latent_channels:
                    raise ValueError("MOSS Edit 需要匹配的 FLUX.2 VAE；请检查 Load VAE。 / Select the matching FLUX.2 VAE in Load VAE.")
                references.append(reference)
        values = {"reference_latents": references}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        return io.NodeOutput(result, positive, negative)


register_patch_format()
NODE_CLASS_MAPPINGS = {"ApplyMossImageEdit": ApplyMossImageEdit}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyMossImageEdit": "Apply MOSS Edit"}
