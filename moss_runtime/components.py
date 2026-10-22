"""Model loading and prompt conditioning for local MOSS editing."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

EDIT_CONTRACT = "ref-prefix-only-t0__target-text-original-moss-v1"


@dataclass(frozen=True)
class Condition:
    embeds: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self):
        if self.embeds.ndim != 3 or not torch.is_floating_point(self.embeds):
            raise ValueError("Condition embeds must be floating [B, S, C] tensors.")
        if min(self.embeds.shape) < 1 or self.attention_mask.shape != self.embeds.shape[:2]:
            raise ValueError("Condition mask must match nonempty [B, S] embeddings.")
        if self.attention_mask.dtype != torch.bool:
            binary = (self.attention_mask == 0) | (self.attention_mask == 1)
            if not bool(binary.all()):
                raise ValueError("Condition attention_mask must be binary.")
        if not bool(self.attention_mask.bool().any(dim=1).all()):
            raise ValueError("Each condition needs at least one valid text token.")

    def to(self, device: torch.device | str) -> Condition:
        return Condition(
            self.embeds.detach().to(device),
            self.attention_mask.detach().to(device=device, dtype=torch.bool),
        )


@dataclass(frozen=True)
class PromptConditions:
    positive: Condition
    negative: Condition | None = None

    def to(self, device: torch.device | str) -> PromptConditions:
        return PromptConditions(
            self.positive.to(device),
            None if self.negative is None else self.negative.to(device),
        )


@dataclass
class TextComponents:
    text_encoder: nn.Module
    tokenizer: Any
    device: torch.device
    text_encoder_device: torch.device
    offload_text_encoder: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _snapshot(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "pipeline_moss_image.py").is_file():
        raise ValueError(f"Expected a complete local MOSS-Image snapshot: {path}")
    return path


def load_moss_module(snapshot_dir: str | Path) -> ModuleType:
    """Import the local official pipeline without loading model weights."""
    source = _snapshot(snapshot_dir) / "pipeline_moss_image.py"
    suffix = hashlib.sha256(str(source).encode() + source.read_bytes()).hexdigest()[:20]
    name = f"_moss_edit_pipeline_{suffix}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import bundled pipeline: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_text_components(snapshot_dir, *, device="cuda", dtype=torch.bfloat16):
    """Load the frozen tokenizer/text encoder; move to GPU when encoding."""
    from transformers import AutoModel, AutoTokenizer

    source = _snapshot(snapshot_dir)
    model_config = json.loads((source / "transformer" / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(source / "tokenizer"), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        str(source / "text_encoder"), dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True,
    ).requires_grad_(False).eval().to("cpu")
    device = torch.device(device)
    return TextComponents(
        encoder, tokenizer, device, device,
        metadata={"text_dim": int(model_config["text_dim"])},
    )


def _condition(value: Condition | Mapping[str, torch.Tensor]) -> Condition:
    if isinstance(value, Condition):
        return value
    if isinstance(value, Mapping):
        return Condition(value["embeds"], value["attention_mask"])
    raise TypeError("Cached conditioning must be a Condition or tensor mapping.")


def _strings(values: str | Sequence[str], batch: int | None = None) -> list[str]:
    if isinstance(values, str):
        result = [values] * (1 if batch is None else batch)
    else:
        result = list(values)
    if not result or not all(isinstance(value, str) for value in result):
        raise ValueError("Prompts must be a string or nonempty sequence of strings.")
    if batch is not None and len(result) != batch:
        raise ValueError("Negative prompts must match the positive batch size.")
    return result


@torch.no_grad()
def encode_prompts(
    components: TextComponents,
    prompts: str | Sequence[str] | None = None,
    *,
    negative_prompts: str | Sequence[str] | None = "",
    max_sequence_length: int = 4096,
    cached: PromptConditions | Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> PromptConditions:
    """Encode positive/negative strings exactly as the official pipeline.

    Each prompt uses an unpadded Qwen forward with thinking enabled; embeddings
    are unnormalized final hidden states. Padding is added only afterwards,
    with a boolean mask. ``negative_prompts=None`` skips negative conditioning.
    ``cached`` may contain either/both positive and negative entries; missing
    entries are encoded. Fully cached calls work without a text encoder.
    """
    if not 1 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be between 1 and 4096.")
    target = components.device if device is None else torch.device(device)
    if isinstance(cached, PromptConditions):
        positive, negative = cached.positive, cached.negative
    elif isinstance(cached, Mapping):
        positive = None if cached.get("positive") is None else _condition(cached["positive"])
        negative = None if cached.get("negative") is None else _condition(cached["negative"])
    elif cached is None:
        positive = negative = None
    else:
        raise TypeError("cached must be PromptConditions or a plain mapping.")
    values = None if prompts is None else _strings(prompts)
    if positive is None and values is None:
        raise ValueError("Positive prompt strings or cached positive conditioning are required.")
    batch = positive.embeds.shape[0] if positive is not None else len(values)
    if values is not None and len(values) != batch:
        raise ValueError("Cached positive conditioning must match the prompt batch size.")
    negatives = None if negative_prompts is None else _strings(negative_prompts, batch)
    if negatives is None:
        negative = None
    needs_encoding = positive is None or (negatives is not None and negative is None)
    encoder = components.text_encoder
    if needs_encoding and (encoder is None or components.tokenizer is None):
        raise ValueError("Uncached prompts require a loaded text encoder and tokenizer.")

    def encode_rows(rows):
        encoded = []
        for value in rows:
            formatted = components.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            inputs = components.tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length,
            ).to(components.text_encoder_device)
            output = encoder(**inputs, use_cache=False, return_dict=True)
            encoded.append(output.last_hidden_state[0].detach().to(target))
        length = max(row.shape[0] for row in encoded)
        embeds = encoded[0].new_zeros(len(encoded), length, encoded[0].shape[-1])
        mask = torch.zeros(len(encoded), length, device=target, dtype=torch.bool)
        for index, row in enumerate(encoded):
            embeds[index, :row.shape[0]] = row
            mask[index, :row.shape[0]] = True
        return Condition(embeds, mask)

    try:
        if needs_encoding:
            encoder.to(components.text_encoder_device).eval()
        if positive is None:
            positive = encode_rows(values)
        if negatives is not None and negative is None:
            negative = encode_rows(negatives)
    finally:
        if needs_encoding and components.offload_text_encoder:
            encoder.to("cpu")
    result = PromptConditions(positive, negative).to(target)
    for value in (result.positive, result.negative):
        if value is not None:
            if value.embeds.shape[0] != batch:
                raise ValueError("Cached negative conditioning must match the positive batch size.")
            if value.embeds.shape[-1] != components.metadata.get("text_dim", value.embeds.shape[-1]):
                raise ValueError("Condition embedding width does not match MOSS text_dim.")
    return result


def torch_dtype(name):
    allowed = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in allowed:
        raise ValueError(f"Unsupported dtype {name!r}; choose float32 or bfloat16.")
    return allowed[name]


def load_transformer(snapshot_dir, device="cuda", dtype=torch.bfloat16):
    source = Path(snapshot_dir).expanduser().resolve(strict=True)
    model = load_moss_module(source).MOSSImageTransformer.from_pretrained(
        str(source / "transformer"), torch_dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    return model.requires_grad_(False).to(device)
