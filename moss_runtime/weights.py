"""Load MOSS editing adapter configurations and weights."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_FORMAT = "moss-edit-lora"
_VERSION = 1
_CONFIG = "adapter_config.json"
_WEIGHTS = "adapter_model.safetensors"


def read_adapter_options(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read architecture options before loading the base model; tensors stay unloaded."""
    config = json.loads((Path(path).expanduser() / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("format") != "moss-edit-lora" or config.get("format_version") != 1:
        raise ValueError("Unsupported LoRA adapter format/version.")
    modules, targets = config.get("modules"), config.get("targets")
    if not isinstance(modules, dict) or not modules or targets != sorted(modules):
        raise ValueError("Adapter must contain consistent, nonempty module/target metadata.")
    keys = ("rank", "alpha", "dropout", "param_dtype")
    first = next(iter(modules.values()))
    if not isinstance(first, dict) or not all(key in first for key in keys):
        raise ValueError("Adapter is missing rank/alpha/dropout/param_dtype.")
    shared = {key: first[key] for key in keys}
    if any(not isinstance(value, dict) or any(value.get(key) != shared[key] for key in keys)
           for value in modules.values()):
        raise ValueError("This sampler requires uniform rank/alpha/dropout/dtype across targets.")
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Adapter metadata must be a JSON object.")
    return {**shared, "targets": targets}, metadata


def _validate_options(rank, alpha, dropout, param_dtype):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer.")
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("LoRA alpha must be finite and positive.")
    if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if param_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("LoRA param_dtype must be torch.float32 or torch.bfloat16.")


class LoRALinear(nn.Module):
    """Frozen original Linear plus an alpha/rank-scaled A/B update.

    The weight/bias properties expose the original Linear for the MOSS model.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0,
                 param_dtype: torch.dtype = torch.float32):
        super().__init__()
        _validate_options(rank, alpha, dropout, param_dtype)
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base.")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = float(dropout)
        factory = {"device": base.weight.device, "dtype": param_dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, **factory), requires_grad=False)
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, **factory), requires_grad=False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base(x)
        x_lora = x.to(self.lora_A.dtype)
        delta = F.linear(F.linear(x_lora, self.lora_A), self.lora_B)
        return original + (delta * self.scaling).to(original.dtype)


def _adapters(model: nn.Module) -> dict[str, LoRALinear]:
    modules = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not modules:
        raise ValueError("Model contains no LoRA adapters.")
    return dict(sorted(modules.items()))


def _structure(model: nn.Module) -> dict[str, Any]:
    result = {}
    for name, module in _adapters(model).items():
        if module.lora_A.dtype != module.lora_B.dtype:
            raise ValueError(f"LoRA A/B dtypes disagree for {name}.")
        _validate_options(module.rank, module.alpha, module.dropout, module.lora_A.dtype)
        result[name] = {
            "rank": module.rank, "alpha": module.alpha, "dropout": module.dropout,
            "in_features": module.in_features, "out_features": module.out_features,
            "param_dtype": str(module.lora_A.dtype).removeprefix("torch."),
        }
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_adapter(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Strictly load into an already-injected model; return user metadata.

    Target names, dimensions, rank, alpha, dropout, and adapter dtypes must
    match exactly. Metadata records provenance without restricting base selection.
    Tensor validation precedes any copy.
    Tensor ``copy_`` increments PyTorch parameter versions; a successful load
    also increments ``model._moss_adapter_version`` for external caches.
    """
    from safetensors import safe_open

    directory = Path(path).expanduser()
    structure = _structure(model)
    config = json.loads((directory / _CONFIG).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("format") != _FORMAT or config.get("format_version") != _VERSION:
        raise ValueError("Unsupported LoRA adapter format/version.")
    if config.get("targets") != list(structure) or config.get("modules") != structure:
        raise ValueError("LoRA structure mismatch: targets/rank/alpha/dropout/dimensions/dtype.")
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("LoRA metadata must be an object.")
    weights = directory / _WEIGHTS
    if config.get("weights_sha256") != _sha256(weights):
        raise ValueError("LoRA weights checksum mismatch; bundle may be incomplete.")
    with safe_open(str(weights), framework="pt", device="cpu") as file:
        embedded = (file.metadata() or {}).get("adapter_config")
        expected_config = {key: value for key, value in config.items() if key != "weights_sha256"}
        if embedded is None or json.loads(embedded) != expected_config:
            raise ValueError("LoRA safetensors/config metadata mismatch.")
        parameters = {
            f"{name}.{key}": getattr(module, key)
            for name, module in _adapters(model).items() for key in ("lora_A", "lora_B")
        }
        if set(file.keys()) != set(parameters):
            raise ValueError("LoRA weight keys mismatch (missing or unexpected tensors).")
        state = {}
        for name, parameter in parameters.items():
            tensor = file.get_tensor(name)
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"LoRA tensor shape/dtype mismatch for {name}.")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Non-finite LoRA tensor: {name}.")
            state[name] = tensor.to(device=parameter.device)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])
    model._moss_adapter_version = int(getattr(model, "_moss_adapter_version", 0)) + 1
    return metadata


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float,
                targets: list[str], param_dtype: torch.dtype) -> None:
    """Install the exact adapter targets listed in the published configuration."""
    model.requires_grad_(False)
    for name in targets:
        base = model.get_submodule(name)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf,
                LoRALinear(base, rank, alpha, dropout, param_dtype))
    model._moss_adapter_version = 0
