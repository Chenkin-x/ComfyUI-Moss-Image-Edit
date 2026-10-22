"""Prepare the frozen model weights used by the editor."""
from torch import nn
from optimum.quanto import quantize, freeze, qint8


def quantize_frozen_linears(model):
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    quantize(model, weights=qint8, include=names)
    freeze(model)
    model.requires_grad_(False)
    return len(names)
