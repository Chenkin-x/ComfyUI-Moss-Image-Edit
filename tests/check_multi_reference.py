"""Portable CPU contract checks; no model downloads or GPU allocations.

Run with ComfyUI's Python: tests/check_multi_reference.py --comfyui /path/to/ComfyUI
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve()))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    import comfy.ops
    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models
    import nodes
    from comfy_extras.nodes_model_patch import ModelPatchLoader
    nodes.NODE_CLASS_MAPPINGS["ModelPatchLoader"] = ModelPatchLoader
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("moss_contract_test", root / "__init__.py",
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plugin = sys.modules[spec.name + ".nodes"]
    apply = plugin.ApplyMossImageEdit
    schema = apply.define_schema()
    assert schema.node_id == "ApplyMossImageEdit"
    assert schema.inputs[-1].id == "reference_images" and schema.inputs[-1].template.min == 0
    assert [output.io_type for output in schema.outputs] == ["MODEL", "CONDITIONING", "CONDITIONING"]
    cfg = dict(dim=64, ffn_dim=128, num_heads=4, num_layers=1, text_dim=32, latent_channels=8, fused_qkv=False)
    config = comfy.supported_models.MOSSImage(cfg)
    config.custom_operations = comfy.ops.disable_weight_init
    base = comfy.model_base.MOSSImage(config, device=torch.device("cpu"))
    patcher = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    target = "layers.0.proj.q"
    weights = {target + ".lora_A": torch.zeros(1, 64), target + ".lora_B": torch.zeros(64, 1)}
    holder = plugin.MossEditWeights(weights, {"modules": {target: {"rank": 1, "alpha": 1, "in_features": 64, "out_features": 64}}})
    patch = comfy.model_patcher.CoreModelPatcher(holder, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    class VAE:
        def __init__(self):
            self.calls = []

        def encode(self, image):
            value = round(float(image[0, 0, 0, 0]), 3)
            self.calls.append((list(image.shape), value))
            return torch.full((1, 8, image.shape[1] // 16, image.shape[2] // 16), value)

    source = torch.full((1, 337, 900, 4), 0.0)
    positive = [[torch.zeros(1, 2, 32), {"existing": "keep", "reference_latents": ["replace"]}]]
    negative = [[torch.zeros(1, 2, 32), {}]]
    vae = VAE()
    result, pos, neg = apply.execute(patcher, patch, vae, positive, negative, source, 512, 1)
    assert vae.calls == [([1, 192, 512, 3], 0.0)]
    assert pos[0][1]["reference_latents"] is neg[0][1]["reference_latents"]
    assert positive[0][1]["reference_latents"] == ["replace"]
    assert "reference_latents" not in negative[0][1]
    assert not patcher.patches and not patcher.object_patches
    vae = VAE()
    # Deliberately scrambled keys, an empty port, differing ratios and a batch.
    extras = {"reference_10": torch.full((1, 48, 32, 3), 0.5), "reference_1": None,
              "reference_2": torch.cat([torch.full((1, 16, 32, 3), 0.3), torch.full((1, 16, 32, 3), 0.4)]),
              "reference_0": torch.full((1, 32, 48, 3), 0.2)}
    result, pos, neg = apply.execute(patcher, patch, vae, positive, negative, source, 512, 1, extras)
    assert [value for _, value in vae.calls] == [0.0, 0.2, 0.3, 0.4, 0.5]
    refs = pos[0][1]["reference_latents"]
    assert len(refs) == 5 and refs is neg[0][1]["reference_latents"]
    assert [round(float(ref[0, 0, 0, 0]), 3) for ref in refs] == [0.0, 0.2, 0.3, 0.4, 0.5]
    assert pos[0][1]["existing"] == "keep"
    before = len(vae.calls)
    bypass = apply.execute(patcher, patch, vae, positive, negative, source, 512, 0, extras)
    assert bypass[0] is patcher and bypass[1] is positive and bypass[2] is negative
    assert len(vae.calls) == before
    print(json.dumps({"passed": True, "single_image_compatible": True, "ordered_images": 5,
                      "numeric_socket_order": True, "batch_frames_are_references": True,
                      "shared_conditions": True, "input_unchanged": True, "strength_zero_bypass": True,
                      "vae_calls": vae.calls}, indent=2))


if __name__ == "__main__":
    main()
