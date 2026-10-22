"""Euler sampling with reference conditioning and classifier-free guidance."""
import math


def sample_latents(editor, scheduler, reference_latents, positive, negative=None, *,
                   height: int, width: int, steps: int = 50, cfg: float = 4,
                   seed: int = 42, device="cuda", latent_channels: int = 128,
                   use_kv_cache: bool = True, progress=None):
    """Euler sampling with one shared reference cache across steps and CFG text.

    Conditions expose ``embeds`` [1,S,D] and ``attention_mask`` [1,S]. Reference
    tensors are [1,C,h,w]. The wrapper receives normalized t; the scheduler
    receives its original 0..1000 timestep. All CFG/Euler arithmetic is FP32.
    """
    import torch

    if min(height, width, steps, latent_channels) < 1 or height % 16 or width % 16:
        raise ValueError("Invalid latent dimensions/steps; image dimensions must be multiples of 16.")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("cfg must be finite and nonnegative.")
    if cfg != 1 and negative is None:
        raise ValueError("CFG requires negative text conditioning.")
    if not reference_latents:
        raise ValueError("Editing requires at least one reference latent.")
    if getattr(scheduler.config, "num_train_timesteps", 1000) != 1000:
        raise ValueError("The edit wrapper requires the official 1000-timestep convention.")
    device = torch.device(device)
    references = [reference.detach().to(device) for reference in reference_latents]
    positive = positive.to(device)
    negative = None if negative is None else negative.to(device)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, height // 16, width // 16),
        generator=generator, device="cpu", dtype=torch.float32,
    ).to(device)
    editor.eval()
    with torch.no_grad():
        cache = editor.prepare_cache(references) if use_kv_cache else None
        scheduler.set_timesteps(steps, device=device)
        total = len(scheduler.timesteps)
        for index, timestep in enumerate(scheduler.timesteps):
            normalized = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(1) / 1000

            def predict(condition):
                return editor(
                    hidden_states=latents, timestep=normalized,
                    encoder_hidden_states=condition.embeds,
                    reference_latents=references,
                    encoder_attention_mask=condition.attention_mask, cache=cache,
                ).float()

            prediction = predict(positive)
            if cfg != 1:
                unconditional = predict(negative)
                prediction = unconditional + cfg * (prediction - unconditional)
            latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0].float()
            if progress is not None:
                progress(index + 1, total)
        if not bool(torch.isfinite(latents).all()):
            raise RuntimeError("Sampling produced non-finite latents.")
    return latents
