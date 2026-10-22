"""[text, ref, target] editing around the unmodified official MOSS transformer.

References use t=0 and attend only to references. Target and text retain the
original joint attention and common timestep, and may also attend references.
Only the reference branch is cached. Latents are already FLUX.2 packed and
batch-normalized [1, 128, H/16, W/16]; public timesteps are normalized to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .components import EDIT_CONTRACT

# Keep the existing checkpoint contract: changing token order with matching RoPE
# and masks is a permutation, not a change to the reference-only attention rule.


def _tensor_stamp(value: torch.Tensor) -> tuple:
    try:
        version = value._version
    except RuntimeError as error:
        raise ValueError(
            "KV cache inputs must have version counters; create reference tensors "
            "under torch.no_grad(), not torch.inference_mode()."
        ) from error
    return (id(value), version, tuple(value.shape), value.dtype, value.device)


@dataclass(frozen=True)
class MossEditCache:
    """An inference-only cache owned by one wrapper and one weight/input version."""

    keys_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    reference_latents: tuple[torch.Tensor, ...] = field(repr=False)
    owner: Any = field(repr=False)
    model_stamp: tuple = field(repr=False)
    reference_stamp: tuple = field(repr=False)
    cache_stamp: tuple = field(repr=False)
    epoch: int = 0
    contract: str = EDIT_CONTRACT


class MossEditModel(nn.Module):
    """Reuse original MOSS modules (including injected LoRA) without new weights.

    ``prepare_cache`` and cache reuse require eval mode and disabled autograd.
    Pass the same reference
    tensors on cached forwards; a different prompt or target geometry is allowed.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self._cache_owner = object()
        self._cache_epoch = 0

    @property
    def config(self):
        return self.base_model.config

    def invalidate_cache(self) -> None:
        """Invalidate caches after changing non-tensor adapter runtime settings."""
        self._cache_epoch += 1

    def _model_stamp(self) -> tuple:
        parameters = tuple((name, _tensor_stamp(value)) for name, value in self.named_parameters())
        buffers = tuple((name, _tensor_stamp(value)) for name, value in self.named_buffers())
        # Standard lightweight LoRA wrappers expose one of these scalar fields.
        settings = tuple(
            (name, attribute, repr(getattr(module, attribute)))
            for name, module in self.named_modules()
            for attribute in ("scaling", "scale", "lora_alpha", "active_adapter", "merged")
            if hasattr(module, attribute)
        )
        device_type = self.base_model.video_embedder.weight.device.type
        autocast = (torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type))
        return parameters, buffers, settings, autocast

    def _require_cache_mode(self) -> None:
        if torch.is_grad_enabled() or any(module.training for module in self.modules()):
            raise RuntimeError("Reference KV caching requires model.eval() and torch.no_grad().")

    def _validate_latent(self, value: torch.Tensor, name: str) -> None:
        if (value.ndim != 4 or value.shape[0] != 1
                or value.shape[1] != self.config.latent_channels
                or min(value.shape[2:]) < 1 or not torch.is_floating_point(value)):
            raise ValueError(f"{name} must be floating [1, {self.config.latent_channels}, H, W].")
        if value.device != self.base_model.video_embedder.weight.device:
            raise ValueError(f"{name} and model must be on the same device.")

    def _references(self, reference_latents: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if not isinstance(reference_latents, (list, tuple)):
            raise ValueError("reference_latents must be a list/tuple of [1,C,H,W] tensors.")
        refs = tuple(reference_latents)
        for index, value in enumerate(refs):
            self._validate_latent(value, f"reference_latents[{index}]")
        return refs

    def _time(self, timestep: torch.Tensor | float, device: torch.device):
        model = self.base_model
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
        if times.numel() != 1 or not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any()):
            raise ValueError("timestep must be one finite normalized noise level in [0,1].")
        times = times * 1000
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        sinusoid = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = model.time_embedding(sinusoid.to(model.time_embedding[0].weight.dtype)).float()
        mod = model.shared_block_modulation(time.to(model.shared_block_modulation.weight.dtype)).view(1, 6, -1)
        head = model.video_head_modulation(time.to(model.video_head_modulation.weight.dtype)).view(1, 2, -1)
        return mod, head

    def _reference_inputs(self, refs: tuple[torch.Tensor, ...]):
        model = self.base_model
        tokens, ropes = [], []
        for index, latent in enumerate(refs):
            height, width = latent.shape[-2:]
            value = latent.flatten(2).transpose(1, 2)
            value = model.video_embedder(value.to(model.video_embedder.weight.dtype).contiguous())
            h, w = torch.meshgrid(
                torch.arange(height, device=value.device), torch.arange(width, device=value.device), indexing="ij"
            )
            coordinates = torch.stack((torch.full_like(h, index + 1), torch.zeros_like(h), h, w), -1).flatten(0, 1).float()
            coordinates[:, 2] -= (height - 1) / 2
            coordinates[:, 3] -= (width - 1) / 2
            phase = coordinates.unsqueeze(-1) * model.rope.bands.float()
            rope = torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)
            tokens.append(value)
            ropes.append(rope.to(value.dtype))
        return torch.cat(tokens, 1), torch.cat(ropes, 1)

    @staticmethod
    def _rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        sin, cos = rope.chunk(2, -1)
        sin = torch.cat((sin, sin), -1).unsqueeze(-2)
        cos = torch.cat((cos, cos), -1).unsqueeze(-2)
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat((-b, a), -1) * sin

    def _qkv(self, layer: nn.Module, x: torch.Tensor, rope: torch.Tensor, mod: torch.Tensor):
        shift, scale = mod[:, 0], mod[:, 1]
        y = layer.norm1(x) * (1 + scale[:, None]) + shift[:, None]
        y = y.to(layer.proj["q"].weight.dtype)
        shape = (*y.shape[:2], layer.heads, y.shape[-1] // layer.heads)
        q = layer.q_norm(layer.proj["q"](y).reshape(shape))
        k = layer.k_norm(layer.proj["k"](y).reshape(shape))
        v = layer.proj["v"](y).reshape(shape).transpose(1, 2)
        q = self._rotate(q, rope).transpose(1, 2).to(v.dtype)
        k = self._rotate(k, rope).transpose(1, 2).to(v.dtype)
        return q, k, v

    @staticmethod
    def _finish(layer: nn.Module, x: torch.Tensor, attention: torch.Tensor, mod: torch.Tensor):
        attention = attention.transpose(1, 2).flatten(2)
        x = x + layer.proj["o"](attention) * mod[:, 2, None]
        y = layer.norm2(x) * (1 + mod[:, 4, None]) + mod[:, 3, None]
        hidden, gate = layer.proj["ffn_up_gate"](y.to(layer.proj["q"].weight.dtype)).chunk(2, -1)
        activated = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
        return x + layer.proj["ffn_down"](activated) * mod[:, 5, None]

    def _reference_block(self, layer, prefix, rope, mod):
        q, k, v = self._qkv(layer, prefix, rope, mod)
        result = self._finish(layer, prefix, F.scaled_dot_product_attention(q, k, v), mod)
        return result, (k, v)

    def _body_block(self, layer, body, rope, mod, reference_kv=None, text_length=0):
        q, k, v = self._qkv(layer, body, rope, mod)
        if reference_kv is not None:
            # Body queries are [text, target]; their keys are [text, ref, target].
            k = torch.cat((k[:, :, :text_length], reference_kv[0], k[:, :, text_length:]), 2)
            v = torch.cat((v[:, :, :text_length], reference_kv[1], v[:, :, text_length:]), 2)
        return self._finish(layer, body, F.scaled_dot_product_attention(q, k, v), mod)

    def _joint_block(self, layer, prefix, body, prefix_rope, body_rope, prefix_mod, body_mod, text_length):
        prefix, kv = self._reference_block(layer, prefix, prefix_rope, prefix_mod)
        return prefix, self._body_block(layer, body, body_rope, body_mod, kv, text_length)

    def prepare_cache(self, reference_latents: Sequence[torch.Tensor]) -> MossEditCache:
        """Prefill references once; reusable across timesteps, CFG prompts and sizes."""
        self._require_cache_mode()
        refs = self._references(reference_latents)
        if not refs:
            raise ValueError("prepare_cache requires at least one reference image.")
        reference_stamp = tuple(_tensor_stamp(value) for value in refs)
        prefix, rope = self._reference_inputs(refs)
        mod, _ = self._time(0.0, prefix.device)
        keys_values = []
        for layer in self.base_model.layers:
            prefix, kv = self._reference_block(layer, prefix, rope, mod)
            keys_values.append(kv)
        keys_values = tuple(keys_values)
        cache_stamp = tuple((_tensor_stamp(k), _tensor_stamp(v)) for k, v in keys_values)
        return MossEditCache(keys_values, refs, self._cache_owner, self._model_stamp(),
                             reference_stamp, cache_stamp, self._cache_epoch)

    def _validate_cache(self, cache: MossEditCache, refs: tuple[torch.Tensor, ...]):
        self._require_cache_mode()
        if not isinstance(cache, MossEditCache) or cache.owner is not self._cache_owner or cache.contract != EDIT_CONTRACT:
            raise ValueError("KV cache belongs to a different model or edit contract.")
        if cache.epoch != self._cache_epoch or cache.model_stamp != self._model_stamp():
            raise ValueError("KV cache is stale: model weights, dtype/device or adapter settings changed.")
        if cache.reference_stamp != tuple(_tensor_stamp(value) for value in refs):
            raise ValueError("KV cache reference tensors changed; prepare a new cache.")
        if cache.cache_stamp != tuple((_tensor_stamp(k), _tensor_stamp(v)) for k, v in cache.keys_values):
            raise ValueError("KV cache tensors were modified.")

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor | float,
                encoder_hidden_states: torch.Tensor, reference_latents: Sequence[torch.Tensor] = (),
                encoder_attention_mask: torch.Tensor | None = None, cache: MossEditCache | None = None) -> torch.Tensor:
        self._validate_latent(hidden_states, "hidden_states")
        refs = self._references(reference_latents)
        model = self.base_model
        context = encoder_hidden_states
        if (context.ndim != 3 or context.shape[0] != 1 or context.shape[1] < 1
                or context.shape[2] != self.config.text_dim or context.device != hidden_states.device):
            raise ValueError("encoder_hidden_states must be [1,S,text_dim] on the model device.")
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask
            if (mask.shape != context.shape[:2] or mask.device != context.device
                    or not bool(((mask == 0) | (mask == 1)).all())):
                raise ValueError("encoder_attention_mask must be a binary [1,S] tensor on the context device.")
            context = context[:, mask[0].bool()]
        if context.shape[1] == 0:
            raise ValueError("At least one valid text token is required.")
        if cache is not None:
            self._validate_cache(cache, refs)
        height, width = hidden_states.shape[-2:]
        video = hidden_states.flatten(2).transpose(1, 2)
        video = model.video_embedder(video.to(model.video_embedder.weight.dtype).contiguous())
        rope = model.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
        text = model.text_embedder(context, rope[:, height * width:])
        text_length = text.shape[1]
        body = torch.cat((text, video), 1)
        # Preserve each token's original spatial/text coordinates when reordering.
        rope = torch.cat((rope[:, height * width:], rope[:, :height * width]), 1)
        mod, head = self._time(timestep, hidden_states.device)
        if refs and cache is None:
            prefix, prefix_rope = self._reference_inputs(refs)
            prefix_mod, _ = self._time(0.0, hidden_states.device)
        for index, layer in enumerate(model.layers):
            if cache is not None:
                body = self._body_block(layer, body, rope, mod, cache.keys_values[index], text_length)
            elif refs:
                prefix, body = self._joint_block(
                    layer, prefix, body, prefix_rope, rope, prefix_mod, mod, text_length,
                )
            else:
                body = self._body_block(layer, body, rope, mod)
        output = model.final_norm_video(body[:, text_length:].to(model.final_norm_video.weight.dtype))
        output = output * (1 + head[:, 1, None]) + head[:, 0, None]
        output = model.final_linear_video(output.to(model.final_linear_video.weight.dtype))
        return output.transpose(1, 2).reshape(1, -1, height, width)
