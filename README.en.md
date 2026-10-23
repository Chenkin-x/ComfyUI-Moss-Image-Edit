# MOSS Image Edit for ComfyUI

[简体中文](README.md) · [Edit examples](https://github.com/Chenkin-x/Moss-Image-Edit)

Use MOSS Edit with native model loaders and KSampler. **Apply MOSS Edit** accepts the model, patch, VAE, positive/negative conditioning a source image and optional additional reference images, and returns `MODEL / positive / negative`. Reference conditioning is encoded once and shared automatically.

## Standard workflows: JSON downloads and full canvas previews

**Start with SFT single-image editing.** Download the JSON and drop it into ComfyUI. Complete the environment check below first, then select your four model files and upload the input images.

| Example | Importable workflow JSON | Full canvas PNG |
|---|---|---|
| **SFT single image (recommended starting point)** | [Download JSON](workflows/moss_edit_single_reference.json) | [View / download PNG](examples/moss_edit_single_reference.png) |
| Base BF16 comparison (quality needs evaluation) | [Download JSON](workflows/moss_edit_base_bf16.json) | [View / download PNG](examples/moss_edit_base_bf16.png) |
| Multi-image editing (implemented; current LoRA quality needs evaluation) | [Download JSON](workflows/moss_edit_multi_reference.json) | [View / download PNG](examples/moss_edit_multi_reference.png) |

[![Complete SFT single-image workflow](examples/moss_edit_single_reference.png)](examples/moss_edit_single_reference.png)

**[Open examples: all three previews, input images and instructions →](examples/README.en.md)**

On the JSON page, click **Download raw file**, not Save webpage. Each PNG also embeds its matching workflow and can be downloaded at original resolution and dropped into ComfyUI. The `.api.json` files are for API requests, not canvas imports. These workflows require the native node implementation in this branch.

## Installation

Use native MOSS Image, Qwen3.5 TE and support for your model's quantization format. First confirm the base model loads in an official workflow. Native MOSS was merged on October 16, 2026, in `e087d8f6`; `v0.41.3` does not include it ([upstream guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)). **This plugin's tested baseline is v0.42.3.** The merge commit is a prerequisite, not a claim that every early revision supports this Edit implementation; the preflight also checks the native APIs it uses.

| Component | Requirement / tested scope |
|---|---|
| Core | Native MOSS, Qwen3.5 TE, MODEL_PATCH, ModelPatcher, prefetch and workflow nodes; v0.42.3 is the tested starting point |
| Python | Follow your core; v0.42.3 declares `>=3.10`, tested locally with 3.12.10 |
| Packages | Follow your **installed core's** requirements.txt, not a second plugin lock. The baseline declares Transformers `>=4.50.3` and safetensors `>=0.4.2`; all older combinations have not been tested |
| Torch / GPU | Keep a working native MOSS environment; tested with Torch 2.9.1+cu130. This plugin does not replace Torch, CUDA, drivers or quantization backends |
| Weights | Prepare the four model types below. Preflight does not load weights or verify available VRAM |

Cross-machine checks on October 23, 2026 (both cores v0.42.3):

| Environment | Verified execution |
|---|---|
| Windows / RTX 4060 Laptop 8GB / Python 3.12.10 / Torch 2.9.1+cu130 / Transformers 5.5.3 | SFT INT8 + TE INT8, single and two-image outputs; Edge desktop dynamic socket connection/disconnection |
| Linux / RTX 5090 32GB / Python 3.14.6 / Torch 2.11.0+cu130 / Transformers 4.57.1 | SFT BF16 + TE BF16, 512 single/two-image outputs and 1024×1024, 30-step two-image output |

These are tested combinations, not an exhaustive version range. **Transformers 5.x is unnecessary in the tested native setup.** Preflight rejected an old v0.35.0 core. Remote testing used a separate core and isolated dependency directory; the shared environment's package inventory was unchanged.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit.git
```

After cloning, run **install.cmd** in the plugin directory on Windows. It checks adjacent launcher/portable Python, `.venv` and `venv`. Missing or ambiguous interpreters stop the script; it never falls back to system Python. For a custom launcher interpreter, specify its exact path:

```bat
install.cmd "E:\AI\ComfyUI-aki-v2\python\python.exe"
```

For Linux/macOS, source installs or a saved report, use **the same Python that starts ComfyUI**:

```bash
python install.py --comfyui /path/to/ComfyUI --report moss-install-report.json
```

Paths are examples. Normal ComfyUI-Manager installation invokes install.py, but manual clones, skipped-dependency modes and other managers may not; run it manually when needed. No empty requirements.txt is shipped: the locally inspected Manager enters a package-repair branch even for an empty requirements file, so omitting it avoids that trigger. This plugin cannot control other manager/launcher repair policies.

On success, restart ComfyUI, refresh the page and import the workflow. This native implementation has **no additional Python dependencies**: no separate environment, Diffusers, Accelerate, Optimum Quanto or forced Transformers 5.x upgrade.

### Diagnostics and environment preservation

The script is read-only: no pip execution, core changes, package replacement or model downloads. Failure leaves the environment unchanged; it does not attempt a destructive install followed by rollback.

1. Show interpreter/core versions and check native MOSS and workflow APIs.
2. Compare installed packages with the current core's requirements, listing missing packages and required/installed versions. Core dependencies marked nonessential are warnings.
3. Report existing package metadata conflicts separately. Historical conflicts from other nodes are not repaired and do not alone fail the MOSS check.
4. Import native modules in a separate CPU process without loading weights or other custom nodes; display actual import failures.
5. Exit `0` for pass (possibly with warnings), `2` for insufficient dependencies/APIs, `3` for an incomplete check. The CMD window stays open.

**Changes to core dependencies remain the user's responsibility through their launcher or environment-maintenance process.** This release has no plugin-specific packages to install automatically. It never upgrades/downgrades/removes existing packages or runs a whole-environment pip upgrade; the checker cannot repair a missing core dependency for you.

| Diagnostic | Next step |
|---|---|
| Wrong Python or missing core | Use the launcher's actual interpreter and pass the core directory with `--comfyui` |
| Missing native MOSS / TE / patch API | Update core and its matching requirements through your normal launcher; a pip package alone cannot add core support |
| Core requires X, installed Y / missing | Review other nodes' constraints, then install/update that dependency yourself and rerun |
| Package A requires B, installed C | Check whether a newer A supports B before choosing an update; the script does not uninstall A |
| DLL or import failure | Inspect the named module/error and repair through your environment owner; Torch is not automatically reinstalled |
| Preflight passes but inference fails | Check weights, quantization, workflow selections, GPU/VRAM and execution logs separately |

Optional reports contain local paths, package versions and import logs. Remove private paths before sharing if needed. Reuse existing weights or download missing files from the sources below. Old Diffusers packages are left in place because other nodes may use them.

## Model files

| File | Directory | Native loader |
|---|---|---|
| ComfyUI-format MOSS Image SFT / Base | `models/diffusion_models/` | Load Diffusion Model |
| Qwen3.5-9B text encoder | `models/text_encoders/` | Load CLIP, type `moss_image` |
| Matching FLUX.2 VAE | `models/vae/` | Load VAE |
| MOSS Edit `adapter_model.safetensors` | `models/model_patches/` | Load Model Patch |

Download the edit patch from [chinoll/Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit). You may rename it to `moss_image_edit.safetensors`. The published file embeds its `adapter_config`; a sidecar JSON is unnecessary. Preserve the embedded metadata.

For base/TE/VAE download links and native formats, see the [official MOSS ComfyUI guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md). Existing weights and ComfyUI's extra model paths can be reused.

Reuse existing ComfyUI single-file models. A full Diffusers snapshot and INT8-to-BF16 conversion are unnecessary. BF16 and supported native INT8 formats use ComfyUI's loading, quantization and offloading.

Start with **SFT + Edit patch**. Base is structurally compatible, but the patch was trained against SFT; evaluate Base editing quality independently.

## Workflows

- [Single reference](workflows/moss_edit_single_reference.json)
- [Base BF16 comparison](workflows/moss_edit_base_bf16.json)
- [Multi-image example (evaluate adapter quality)](workflows/moss_edit_multi_reference.json)
- Matching `.api.json` files are provided for API use.

Drag the JSON into ComfyUI, select your local model filenames and upload references. Example filenames and `moss/` subdirectories reflect the tested installation, not mandatory naming.

The multi-image example uses `example_reference.png` as a placeholder for the second image. Upload/select your image if the frontend reports missing media.

1. Connect the native model, patch and VAE loaders and both CLIP Text Encode outputs to Apply MOSS Edit.
2. Connect the original image directly to Apply. `reference_max_edge=512` preserves the publisher's default: shrink the long edge only, align dimensions to 16, without cropping. Encoding happens once, replacing existing references on both output conditions.
3. Branch the original image into **Scale Image to Total Pixels → Get Image Size → Empty Flux 2 Latent**. Set `megapixels=1.0`, `resolution_steps=16`. This controls output dimensions; the 1MP image is not used for reference encoding.
4. Route Apply's model output through the visible **SkipLayerGuidanceDiTSimple** node to KSampler. Connect Apply's conditioning outputs directly to KSampler, then VAE Decode and Save Image.

Start with a `512`-pixel reference long-edge limit, approximately `1MP` output, strength `1`, `Euler / normal`, `30 steps`, `CFG 4`, `denoise 1`. The native pixel node uses 1024×1024 pixels per MP; aspect-preserving alignment introduces small rounding differences. **Pixel dimensions must be multiples of 16**; latent grid dimensions need not themselves be multiples of 16.

**Multi-image inputs participate in inference.** The existing `image` input is labeled source image and remains first. Optional native Autogrow IMAGE sockets accept additional references in numeric `reference_0, reference_1, reference_2…` order. Each image is independently resized and encoded; both conditions share the same ordered reference sequence. Leaving all additional sockets disconnected preserves single-image workflows.

Current multi-image quality depends on adapter training. Future adapters using the same format and edit contract can replace the patch file; arbitrary future contracts are not guaranteed. Source/reference labels describe the UI ordering convention: the backend adds no separate semantic role labels.

ComfyUI currently allows up to 100 additional Autogrow sockets. This is a framework limit, not a model capability or recommended count. More images cost memory and compute. An IMAGE batch is expanded in frame order into multiple references, **not independent editing jobs**. One Load Image per socket is the clearest workflow.

 A higher `reference_max_edge` is available for experiments; successful execution alone does not establish editing quality. A 1MP output is independent of a 1MP reference.

Different reference and output scales can affect composition. In one local same-seed example, a 512 reference with 1MP output changed the composition, while a 1MP reference with 1MP output stayed closer to the original. This is not a general quality conclusion. Compare 512-reference / 0.25MP-output first when fidelity changes, then test higher reference limits. Raising the limit does not upscale a small source image.

## Shift and SLG

Native ComfyUI detects MOSS and configures **inference_shift=5.0, multiplier=1.0** automatically. These are model defaults, not resolution-adaptive estimates. The standard workflow needs no extra sampling configuration node. To override shift, insert **ModelSamplingAuraFlow** before SLG, not ModelSamplingSD3. See the [official integration guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md).

SLG is explicit, not hidden in Apply: `single_layers=22` (zero-indexed), empty `double_layers`, start `0`, end `1`. This follows the [base model recommendation](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md), not a mandatory Edit requirement or a quality guarantee. Clear `single_layers` or bypass the node to compare with SLG disabled.

For limited VRAM, reduce the output branch to `0.25MP`. Load CLIP's `cpu` device option reduces text-encoder VRAM at the cost of speed. Sample from an empty FLUX.2 latent. Native scheduling does not reproduce the old pipeline's 1000-step grid rounding, so matching seeds do not guarantee pixel-identical results.

## Implementation

Edit weights use native `ModelPatcher.add_patches` on a model clone. No pre-merge or checkpoint rewrite is required. Fused Q/K/V models use native LoRA slice mappings. References use t=0 and attend only to references; text and target retain MOSS joint attention. Native blocks, attention backends, prefetch and memory management are reused.

ModelPatchLoader currently lacks a third-party format registry. At startup this plugin extends the registered native loader only for embedded MOSS Edit metadata, delegating other files to the original implementation. It does not edit core files. This integration point needs maintenance when upstream loader APIs change.

Import the new workflows to replace the old `MOSS_EDIT_MODEL` interface. Existing model files are untouched. Reference KV caching has not yet been ported; the native implementation recomputes references each forward pass.
