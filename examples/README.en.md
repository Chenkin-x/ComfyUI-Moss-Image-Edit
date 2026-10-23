# Standard workflows

[简体中文](README.md) · **English** · [Back to the project guide](../README.en.md)

Download a JSON or an original workflow PNG and drop it into ComfyUI. Install this branch's nodes, pass `install.py`, and restart ComfyUI first; the tested core is v0.42.3. Select your local diffusion model, TE, VAE and Edit patch. Put the patch in `models/model_patches/`; no pre-merging is required.

- Input examples: [example.png](example.png), [example_reference.png](example_reference.png). Download and upload them in the matching Load Image nodes; JSON import does not upload media. The first is ComfyUI's example image; the second is a generated red-dress reference.
- Defaults: reference long edge 512, output about 1MP, pixel dimensions aligned to 16, 30 steps, CFG 4, Euler / normal. Native MOSS sets shift 5.0; SLG layer 22 is visible in the graph.
- SFT is the starting point for single-image editing; Base is a comparison. Multi-image execution works, but current LoRA quality needs evaluation.
- Model filenames are examples; select the files available locally. You do not need to use the same subdirectories.
- [Read the full workflow settings](../README.en.md#workflows).

## SFT single-image editing

[Workflow JSON](../workflows/moss_edit_single_reference.json) · [API JSON](../workflows/moss_edit_single_reference.api.json) · [Full PNG](moss_edit_single_reference.png)

[![SFT single-image editing](moss_edit_single_reference.png)](moss_edit_single_reference.png)

## Base comparison

[Workflow JSON](../workflows/moss_edit_base_bf16.json) · [API JSON](../workflows/moss_edit_base_bf16.api.json) · [Full PNG](moss_edit_base_bf16.png)

[![Base comparison](moss_edit_base_bf16.png)](moss_edit_base_bf16.png)

## Multi-image editing

[Workflow JSON](../workflows/moss_edit_multi_reference.json) · [API JSON](../workflows/moss_edit_multi_reference.api.json) · [Full PNG](moss_edit_multi_reference.png)

[![Multi-image editing](moss_edit_multi_reference.png)](moss_edit_multi_reference.png)
