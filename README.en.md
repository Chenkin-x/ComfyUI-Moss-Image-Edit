<p align="center">
  <a href="README.md">简体中文</a> &nbsp; · &nbsp; <strong>English</strong>
</p>

<p align="center">
  <img src="assets/moss-image-edit-banner-cases.png" alt="MOSS Image Edit — Edit images with words" width="1280">
</p>

<h1 align="center">MOSS Image Edit for ComfyUI</h1>

<p align="center">
  <strong>Upload references. Describe the change. Generate.</strong><br>
  Use the MOSS Image Edit <strong>image-editing LoRA</strong> in ComfyUI to edit expressions, outfits, objects, and scenes with one to four reference images.<br>
  Load the MOSS-Image Base or SFT model together with this project's LoRA weights.
</p>

<p align="center">
  <a href="#install">Install</a> &nbsp; · &nbsp;
  <a href="#download-the-models">Models</a> &nbsp; · &nbsp;
  <a href="#run-a-workflow">Workflows</a> &nbsp; · &nbsp;
  <a href="https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/README.en.md#editing-showcase">Editing showcase</a>
</p>

## Install

If you installed the previous nodes, move `custom_nodes/Moss-Image-Edit` outside `custom_nodes` to avoid duplicate registration; keep your model folders in place.

Run these commands in **the Python environment used by ComfyUI**:

```bash
cd /path/to/ComfyUI
git clone https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit.git custom_nodes/ComfyUI-Moss-Image-Edit
python -m pip install -r custom_nodes/ComfyUI-Moss-Image-Edit/requirements.txt
```

Restart ComfyUI and refresh the page. The model loader and editing node appear under `MOSS Image Edit`. For the Windows portable edition, install dependencies with its bundled `python_embeded/python.exe`; see the [official ComfyUI installation guide](https://docs.comfy.org/installation/install_custom_node).

## Download the models

Both [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) and [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) are supported. **Choose either base model** and download this project's [editing LoRA weights](https://huggingface.co/chinoll/Moss-Image-Edit). The following example uses SFT; run from the ComfyUI root:

```bash
hf download OpenMOSS-Team/MOSS-Image-SFT \
  --local-dir models/moss_image/MOSS-Image-SFT

hf download chinoll/Moss-Image-Edit \
  adapter_config.json adapter_model.safetensors \
  --local-dir models/moss_image/Moss-Image-Edit
```

For Base, use this base-model download command instead:

```bash
hf download OpenMOSS-Team/MOSS-Image-Base \
  --local-dir models/moss_image/MOSS-Image-Base
```

The resulting layout, using SFT as the example, is:

```text
ComfyUI/
├── custom_nodes/ComfyUI-Moss-Image-Edit/
└── models/moss_image/
    ├── MOSS-Image-SFT/
    │   ├── pipeline_moss_image.py
    │   ├── transformer/
    │   ├── text_encoder/
    │   ├── tokenizer/
    │   ├── vae/
    │   └── scheduler/
    └── Moss-Image-Edit/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

The loader's `snapshot_dir` points to the complete base model, and `adapter_dir` points to the LoRA weights. Their defaults match this layout. Relative paths start at `ComfyUI/models`; absolute paths also work.

When using Base, set the loader's `snapshot_dir` to `moss_image/MOSS-Image-Base`.

## Run a workflow

| Task | Workflow to drop into ComfyUI |
|:---|:---|
| Edit one image | [moss_edit_single_reference.json](workflows/moss_edit_single_reference.json) |
| Edit with two references | [moss_edit_multi_reference.json](workflows/moss_edit_multi_reference.json) |

1. Download a workflow JSON and drop it onto the ComfyUI canvas.
2. Upload your reference into each `Load Image` node.
3. Enter the edit in `MOSS Image Edit` → `prompt`, for example `Change the background to a sunny beach, preserving the person.` Leave `negative_prompt` empty if you do not need it.
4. Choose the output `width` and `height`, then click **Run**. `Save Image` previews the result and saves it under `ComfyUI/output/MOSS-Image-Edit/`.

The multi-reference workflow connects two images. Add `Load Image` nodes and connect them to `reference_3` and `reference_4` for three or four references. Use image 1, image 2, etc. in the instruction to identify their roles, for example `Dress the person in image 1 in the outfit from image 2.` Connect one image per input; references can have different aspect ratios.

## Settings

| Setting | Workflow default | Purpose |
|:---|:---|:---|
| `width` / `height` | `512` / `512` | Output dimensions, both multiples of 16. |
| `reference_max_edge` | `512` | Maximum reference long edge; dimensions are aligned to 16 without cropping. |
| `steps` | `30` | Denoising steps. |
| `cfg` | `4.0` | Text guidance strength. |
| `seed` | `42`, `fixed` | Initial noise; keep fixed when comparing settings. |
| `inference_shift` | `5.0` | Sampling time distribution. |

For API clients, use the [single-reference API JSON](workflows/moss_edit_single_reference.api.json) or [multi-reference API JSON](workflows/moss_edit_multi_reference.api.json). Image names refer to files under `ComfyUI/input/`. After replacing them, send the JSON as the `prompt` object in a ComfyUI `/prompt` request.

---

This project is licensed under [GNU GPLv3](LICENSE).
