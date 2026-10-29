"""Reference-prefix editing using native MOSS blocks and attention backends."""
from functools import partial

import torch

import comfy.model_prefetch
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import modulated_rms_norm


def reference_attention(attention, q, k, v, heads, *, prefix_length, previous=None, **kwargs):
    call = partial(previous, attention) if previous is not None else attention
    prefix = call(q[:, :prefix_length], k[:, :prefix_length], v[:, :prefix_length], heads, **kwargs)
    body = call(q[:, prefix_length:], k, v, heads, **kwargs)
    return torch.cat((prefix, body), dim=1)


def edit_forward(model, x, timestep, context, transformer_options=None, reference_latents=(), **kwargs):
    batch, _, height, width = x.shape
    dtype, device = x.dtype, x.device
    options = (transformer_options or {}).copy()
    pe = model.rope(height, width, context.shape[1], device)
    vec = model.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    zero_vec = model.time_embedding(timestep_embedding(torch.zeros_like(timestep), 256).to(dtype))
    body_mod = model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    ref_mod = model.shared_block_modulation(zero_vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = model.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    text = model.text_embedder(context, pe[:, height * width:], options)
    body = torch.cat((model.video_embedder(x.flatten(2).transpose(1, 2)), text), dim=1)
    prefixes, positions = [], []
    for index, ref in enumerate(reference_latents):
        h, w = ref.shape[-2:]
        prefixes.append(model.video_embedder(ref.flatten(2).transpose(1, 2)))
        ids = torch.zeros(h * w, 4, device=device, dtype=torch.float32)
        ids[:, 0] = index + 1
        ids[:, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
        ids[:, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
        if hasattr(model, "pe_embedder"):
            ref_pe = model.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()
        else:
            from comfy.ldm.flux.math import rope
            ref_pe = torch.cat([
                rope(ids.unsqueeze(0)[..., axis], dim, 10000)
                for axis, dim in enumerate(model.rope_axes_dim)
            ], dim=-3).unsqueeze(1).transpose(1, 2).contiguous()
        positions.append(ref_pe)
    prefix_length = sum(value.shape[1] for value in prefixes)
    mod = tuple(torch.cat((ref.expand(-1, prefix_length, -1), target.expand(-1, body.shape[1], -1)), dim=1)
                for ref, target in zip(ref_mod, body_mod))
    x = torch.cat((*prefixes, body), dim=1)
    pe = torch.cat((*positions, pe), dim=1)
    del body, text, prefixes, positions
    options["optimized_attention_override"] = partial(
        reference_attention, prefix_length=prefix_length, previous=options.get("optimized_attention_override"))
    options["total_blocks"] = len(model.layers)
    options["block_type"] = "single"
    replacements = options.get("patches_replace", {}).get("dit", {})
    queue = comfy.model_prefetch.make_prefetch_queue(list(model.layers), device, options)
    comfy.model_prefetch.malloc_graph_begin(device)
    try:
        for index, block in enumerate(model.layers):
            comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype, malloc_scope="block")
            options["block_index"] = index
            if ("single_block", index) in replacements:
                def original_block(args):
                    return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"])}
                x = replacements[("single_block", index)](
                    {"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": options},
                    {"original_block": original_block})["img"]
            else:
                x = block(x, mod, pe, options)
    finally:
        comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()
    target = x[:, prefix_length:prefix_length + height * width]
    out = model.final_linear_video(modulated_rms_norm(target, model.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(batch, model.latent_channels, height, width)
