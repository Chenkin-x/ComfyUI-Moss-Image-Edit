# 标准工作流 / Standard workflows

下载 JSON 或含工作流的原始 PNG，拖入 ComfyUI。请先安装本分支节点、通过 `install.py` 检查，再重启 ComfyUI；已验证核心为 v0.42.3。选择本机的底模、TE、VAE 和 Edit 补丁。补丁放在 `models/model_patches/`，无需预先合并。

Download a JSON or an original workflow PNG and drop it into ComfyUI. Install this branch's nodes, pass `install.py`, and restart ComfyUI first; the tested core is v0.42.3. Select your local diffusion model, TE, VAE and Edit patch. Put the patch in `models/model_patches/`; no pre-merging is required.

- 输入示例 / Input examples: [example.png](example.png), [example_reference.png](example_reference.png). Download and upload them in the matching Load Image nodes; JSON import does not upload media. The first is ComfyUI's example image; the second is a generated red-dress test reference.
- 默认 / Defaults: reference long edge 512, output about 1MP, pixel dimensions aligned to 16, 30 steps, CFG 4, Euler / normal. Native MOSS sets shift 5.0; SLG layer 22 is visible in the graph.
- 单图推荐使用 SFT；Base 仅作为对照。多图接口可实际推理，当前 LoRA 的多图质量仍需评估。SFT is the starting point; Base is a comparison. Multi-image execution works, but current LoRA quality needs evaluation.
- 模型名称是示例，按本地文件重新选择；不需要采用相同子目录。Model filenames are examples; select the files available locally.
- 完整参数说明 / Details: [中文 README](../README.md#完整工作流) · [English README](../README.en.md).

## SFT 单图编辑 / Single-image editing

[工作流 JSON / Workflow JSON](../workflows/moss_edit_single_reference.json) · [API JSON](../workflows/moss_edit_single_reference.api.json) · [完整 PNG / Full PNG](moss_edit_single_reference.png)

[![SFT 单图编辑 / Single-image editing](moss_edit_single_reference.png)](moss_edit_single_reference.png)

## Base BF16 对照 / Comparison

[工作流 JSON / Workflow JSON](../workflows/moss_edit_base_bf16.json) · [API JSON](../workflows/moss_edit_base_bf16.api.json) · [完整 PNG / Full PNG](moss_edit_base_bf16.png)

[![Base BF16 对照 / Comparison](moss_edit_base_bf16.png)](moss_edit_base_bf16.png)

## 多图编辑 / Multi-image editing

[工作流 JSON / Workflow JSON](../workflows/moss_edit_multi_reference.json) · [API JSON](../workflows/moss_edit_multi_reference.api.json) · [完整 PNG / Full PNG](moss_edit_multi_reference.png)

[![多图编辑 / Multi-image editing](moss_edit_multi_reference.png)](moss_edit_multi_reference.png)

