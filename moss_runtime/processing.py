"""Reference-image encoding and result decoding in the MOSS latent space."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EncodingComponents:
    prompt_components: Any
    vae: Any
    snapshot_dir: str


def image_to_tensor(image):
    """PIL RGB -> CPU FP32 [1, 3, H, W], normalized to [-1, 1]."""
    import numpy as np
    import torch

    pixels = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).div_(127.5).sub_(1)


def _vae_placement(vae):
    import torch

    parameter = next(vae.parameters(), None)
    if parameter is not None:
        return parameter.device, parameter.dtype
    return vae.bn.running_mean.device, getattr(vae, "dtype", torch.float32)


def _bn_stats(vae, channels: int, device):
    import torch

    mean = vae.bn.running_mean.to(device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    variance = vae.bn.running_var.to(device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    eps = float(vae.config.batch_norm_eps)
    if mean.numel() != channels or variance.numel() != channels:
        raise ValueError("Packed latent channels must match FLUX.2 batch-normalization statistics.")
    if not math.isfinite(eps) or eps <= 0 or not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("Invalid FLUX.2 batch-normalization statistics.")
    if (variance < 0).any():
        raise ValueError("FLUX.2 running variance must be nonnegative.")
    return mean, (variance + eps).sqrt()


def encode_image(vae, image, *, sample_posterior: bool = False, generator=None):
    """Frozen FLUX.2 encode -> 2x2 channel pack -> BN normalization.

    Input is [B, 3, H, W] in [-1, 1]; result is FP32 [B, 128, H/16, W/16]
    for the official 32-channel VAE. Posterior mode matches Flux2Pipeline.
    """
    import torch

    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must be a [B, 3, H, W] tensor.")
    if not image.is_floating_point() or min(image.shape) < 1 or any(s % 16 for s in image.shape[-2:]):
        raise ValueError("Image dimensions must be nonempty multiples of 16 and pixels floating point.")
    if not torch.isfinite(image).all() or image.min() < -1.001 or image.max() > 1.001:
        raise ValueError("Image pixels must be finite and normalized to [-1, 1].")
    device, dtype = _vae_placement(vae)
    with torch.no_grad():
        vae.eval()
        posterior = vae.encode(image.to(device=device, dtype=dtype)).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if z.ndim != 4 or z.shape[-2] % 2 or z.shape[-1] % 2:
            raise ValueError("VAE latent spatial dimensions must be even for 2x2 packing.")
        batch, channels, height, width = z.shape
        # Official Flux2Pipeline._patchify_latents ordering: c, dy, dx.
        z = z.reshape(batch, channels, height // 2, 2, width // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).reshape(batch, 4 * channels, height // 2, width // 2).float()
        mean, std = _bn_stats(vae, z.shape[1], z.device)
        normalized = (z - mean) / std
        if not torch.isfinite(normalized).all():
            raise ValueError("VAE encoding produced nonfinite latents.")
        return normalized


def decode_latents(vae, latents):
    """Inverse normalized packed latent transform, then frozen VAE decode.

    Returns floating [B, 3, H, W], clamped to [-1, 1] as the official MOSS
    pipeline. Normalization arithmetic is FP32 before conversion to VAE dtype.
    """
    import torch

    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or min(latents.shape) < 1:
        raise ValueError("latents must be a nonempty [B, C, H, W] tensor.")
    if not latents.is_floating_point() or latents.shape[1] % 4 or not torch.isfinite(latents).all():
        raise ValueError("Latents must be finite floating point, with channels divisible by four.")
    device, dtype = _vae_placement(vae)
    with torch.no_grad():
        vae.eval()
        z = latents.to(device=device, dtype=torch.float32)
        mean, std = _bn_stats(vae, z.shape[1], device)
        z = z * std + mean
        batch, channels, height, width = z.shape
        z = z.reshape(batch, channels // 4, 2, 2, height, width)
        z = z.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, 2 * height, 2 * width)
        result = vae.decode(z.to(dtype=dtype), return_dict=False)[0]
        if not torch.isfinite(result).all():
            raise ValueError("VAE decoding produced nonfinite images.")
        return result.clamp(-1, 1)


def load_encoding_components(snapshot_dir, *, device="cuda", dtype=None, vae_dtype=None):
    """Load the frozen Qwen/tokenizer and FLUX.2 VAE for editing."""
    from importlib.metadata import version

    import torch
    from packaging.version import Version

    if Version(version("diffusers")) < Version("0.38.0"):
        raise RuntimeError(
            "MOSS-Image's asymmetric FLUX.2 VAE requires diffusers>=0.38.0; "
            "older versions do not support decoder_block_out_channels."
        )
    from . import components as adapter
    dtype = torch.bfloat16 if dtype is None else dtype
    vae_dtype = torch.float32 if vae_dtype is None else vae_dtype
    source = Path(snapshot_dir).expanduser().resolve(strict=True)
    prompts = adapter.load_text_components(source, device=device, dtype=dtype)
    module = adapter.load_moss_module(source)
    vae = module.AutoencoderKLFlux2.from_pretrained(
        str(source / "vae"), torch_dtype=vae_dtype, local_files_only=True,
    ).requires_grad_(False).eval().to(device)
    return EncodingComponents(prompts, vae, str(source))
