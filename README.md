# MOSS Image Edit for ComfyUI

[English](README.en.md) · [编辑效果](https://github.com/Chenkin-x/Moss-Image-Edit)

> [!IMPORTANT]
> **ComfyUI 官方已支持 MOSS-Image Edit**，详见已合并的 [ComfyUI #17296](https://github.com/Comfy-Org/ComfyUI/pull/17296)。
> 请将 **ComfyUI 本体**更新到包含该改动的最新版本，重启并刷新页面后使用官方节点。在 **Text Encode MOSS-Image Edit** 中，将 `edit_format` 设为 **Moss-Image-Edit**。

**训练代码：** [Chenkin-x/Moss-Image-Edit](https://github.com/Chenkin-x/Moss-Image-Edit) · [训练入口 `train.py`](https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/train.py) · [训练配置 `configs/train.json`](https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/configs/train.json)。

以下是本仓库自定义节点与示例工作流的使用说明。

通过官方加载器和 KSampler 使用 MOSS Edit。**Apply MOSS Edit** 集中接收模型、补丁、VAE、正负条件、待编辑原图和可选的多张附加参考图，输出 `MODEL / positive / negative`，无需手动复制参考 latent 连线。

## 标准工作流：下载 JSON / 查看完整连线

**首次使用推荐 SFT 单图编辑。** 下载 JSON 后拖入 ComfyUI；导入前先完成下方环境检查，导入后选择自己的四类模型并上传输入图。

| 示例 | 可导入的工作流 JSON | 完整画布 PNG |
|---|---|---|
| **SFT 单图编辑（推荐起点）** | [下载 JSON](workflows/moss_edit_single_reference.json) | [查看 / 下载图片](examples/moss_edit_single_reference.png) |
| Base BF16 对照（效果需评估） | [下载 JSON](workflows/moss_edit_base_bf16.json) | [查看 / 下载图片](examples/moss_edit_base_bf16.png) |
| 多图编辑（已接入推理，当前 LoRA 效果需评估） | [下载 JSON](workflows/moss_edit_multi_reference.json) | [查看 / 下载图片](examples/moss_edit_multi_reference.png) |

[![SFT 单图编辑完整工作流](examples/moss_edit_single_reference.png)](examples/moss_edit_single_reference.png)

**[打开示例目录：三套大图、输入图片和使用步骤 →](examples/README.md)**

JSON 页面点击 **Download raw file** 保存，不要保存 GitHub 网页。PNG 同时内嵌对应工作流，也可下载原图后拖入 ComfyUI；`.api.json` 供 API 调用，不是画布导入文件。这些示例需要本分支配套的原生节点实现。

## 安装

先使用已经原生支持 **MOSS Image、Qwen3.5 TE 和对应量化格式**的 ComfyUI，底模应先能在官方工作流中正常加载。原生 MOSS 于 2026-10-16 合入，核心至少需包含 `e087d8f6`；`v0.41.3` 尚不包含支持，见 [官方说明](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)。**本插件实际验证基线为官方 v0.42.3**；合入提交只是 MOSS 的起点，不代表所有早期版本都已验证 Edit 接口。预检还会检查本插件使用的原生接口。

| 项目 | 要求与验证范围 |
|---|---|
| ComfyUI 核心 | 原生 MOSS、Qwen3.5 TE、MODEL_PATCH、ModelPatcher、预取接口及标准工作流节点齐全；推荐以已验证的 v0.42.3 为起点 |
| Python | 遵循所用核心的要求；v0.42.3 声明 `>=3.10`，本机验证 3.12.10，不将所有 Python 版本视为已测试 |
| Python 包 | 采用**本机当前核心**的 `requirements.txt`，不复制一份容易过期的插件版本锁。该基线声明 Transformers `>=4.50.3`、safetensors `>=0.4.2`，这不是对所有旧版本组合的实测承诺 |
| PyTorch / GPU | 沿用能运行原生 MOSS 的环境；本机验证 Torch 2.9.1+cu130。插件不负责替换 Torch、CUDA、驱动或量化后端 |
| 模型 | 事先准备下表四类权重；文件格式需受所用核心支持。安装检查不加载模型，不证明文件正确或显存够用 |

2026-10-23 跨机器实测（均为核心 v0.42.3）：

| 环境 | 模型与验证 |
|---|---|
| Windows / RTX 4060 Laptop 8GB / Python 3.12.10 / Torch 2.9.1+cu130 / Transformers 5.5.3 | SFT INT8 + TE INT8，单图与双图均实际出图；Edge 桌面画布拖线、动态端口增删通过 |
| Linux / RTX 5090 32GB / Python 3.14.6 / Torch 2.11.0+cu130 / Transformers 4.57.1 | SFT BF16 + TE BF16，512 单图、512 双图、1024×1024/30步双图均实际出图 |

这些是已验证组合，不表示所有中间版本都已测试；**至少在上述原生环境中无需 Transformers 5.x**。远端旧核心 v0.35.0 被预检明确拒绝；测试使用独立核心和隔离依赖目录，共享环境包清单前后未变。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit.git
```

克隆后，在插件目录双击 **`install.cmd`** 检查。它只查找相邻的秋叶/便携 Python 或 `.venv` / `venv`；找不到或找到多个时会停止，不回退到系统 Python。启动器使用自定义 Python 时，请显式指定实际解释器：

```bat
install.cmd "E:\AI\ComfyUI-aki-v2\python\python.exe"
```

Linux/macOS、源码安装或需要保存诊断报告时，用**启动 ComfyUI 的同一个 Python**运行：

```bash
python install.py --comfyui /path/to/ComfyUI --report moss-install-report.json
```

路径仅为示例。ComfyUI-Manager 的常规安装流程会调用 `install.py`；不同管理器、跳过依赖模式和手动克隆不保证执行，必要时手动运行。当前版有意不附空的 `requirements.txt`：本机 Manager 即使读到空依赖文件也会进入包修复分支，去掉它可避免这条不必要的触发路径。Manager 或启动器自身其他依赖修复策略不由本插件控制。

检查通过后，重启 ComfyUI 并刷新页面、导入标准工作流。这一原生实现**没有插件专属 Python 依赖**，正常安装不需要下载或升级包，不另建环境，不要求额外安装 Diffusers、Accelerate、Optimum Quanto 或指定 Transformers 5.x。

### 安装诊断与环境保护

`install.py` 是只读预检，不在启动时执行 pip，不修改核心、现有包或模型文件。因此没有“先覆盖再尝试退回”的步骤，检查失败时原环境保持不变。

1. 打印实际 Python 路径、Python 和核心版本；检查原生 MOSS 及工作流接口。
2. 对照当前核心声明，逐项列出**缺失包、已安装版本和所需版本范围**。核心标为非必要的依赖单独警告。
3. 列出当前环境已存在的包依赖声明冲突。其他插件的历史冲突不自动修复，也不单独作为 MOSS 不可用的结论。
4. 在独立 CPU 进程导入所需原生模块，不加载模型、不执行其他自定义节点。导入失败会显示具体异常。
5. 返回 `0` 表示预检通过（仍可能有历史警告），`2` 表示依赖/接口不充分，`3` 表示检查未完成。CMD 窗口保留结果，失败明确提示“节点可能无法正常使用”。

**需要升级的是 ComfyUI 自己的依赖时，由用户通过原启动器或自己的环境维护流程处理。** 本插件不会为满足核心要求强行升级、降级或卸载已有包，也不会给整个环境执行 `pip install -U`。这版没有额外包需要自动补装；不把“安装 Edit 节点”变成“替所有插件更新环境”。这也意味着预检只提示和核验，不能自动修好一个缺少核心依赖的安装。

| 提示 | 处理方式 |
|---|---|
| 找不到 ComfyUI / Python 路径不对 | 核对启动器实际 Python；用 `--comfyui` 指定核心根目录 |
| 缺少 MOSS / TE / ModelPatchLoader 等接口 | 通过原启动器更新核心及其配套依赖，不能只安装某个 pip 包代替核心更新 |
| `ComfyUI requires X... installed Y` / `not installed` | 按报告核对该核心所需的 X；检查其他插件约束后由用户补装或升级，再重跑检查 |
| `A requires B... installed C` | 属于已安装包的声明冲突；先确认 A 的新版是否已支持 B，再由用户决定更新 A 或保留现状。脚本不卸载 A |
| DLL / 导入异常 | 查看打印的具体模块和异常；使用原启动器修复对应运行库或包，检查不会自动重装 Torch |
| 通过预检但出图失败 | 再核对四类权重、量化格式、工作流选择、GPU/显存和采样报错；预检通过不是推理成功保证 |

报告含本机路径、包版本及导入日志，分享前可删除不需要的路径信息。模型下载不属于包依赖修复：已有权重直接沿用，缺少的由用户按下表来源选择下载，不后台下载整套模型。安装器也不会删除旧 Diffusers 环境中的包，因为其他节点可能仍在使用。

## 模型位置

| 文件 | 目录 | 官方节点 |
|---|---|---|
| ComfyUI 支持的 MOSS Image SFT / Base 单文件底模 | `models/diffusion_models/` | Load Diffusion Model |
| Qwen3.5-9B 文本编码器 | `models/text_encoders/` | Load CLIP，type 选 `moss_image` |
| 对应的 FLUX.2 VAE | `models/vae/` | Load VAE |
| MOSS Edit `adapter_model.safetensors` | `models/model_patches/` | Load Model Patch |

编辑补丁来自 [chinoll/Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit)。可以将 `adapter_model.safetensors` 重命名为 `moss_image_edit.safetensors`。已发布文件内含 `adapter_config` 元数据，无需旁边再放 JSON。不要移除该元数据。

底模、TE、VAE 的下载入口和原生格式选择见 [MOSS 官方 ComfyUI 指南](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)。已有相同权重不必重复下载，额外模型路径可继续使用 ComfyUI 自己的配置。

底模使用官方加载器支持的 ComfyUI 单文件格式，不再要求下载一整套 Diffusers 目录，也不需要将 INT8 模型先转换成 BF16。已有模型可直接沿用。BF16 与 ComfyUI 原生支持的 INT8 都走官方加载、量化与显存管理。

推荐先用 **SFT + Edit patch**。Base 的结构可以加载同一补丁，但该补丁的训练基线是 SFT，Base 的编辑效果需要单独评估。不要把“能加载”当作效果保证。

## 完整工作流

- [单图编辑](workflows/moss_edit_single_reference.json)
- [Base BF16 对照](workflows/moss_edit_base_bf16.json)
- [多图输入示例（效果需评估）](workflows/moss_edit_multi_reference.json)
- 对应 `.api.json` 文件用于 API 调用。

将 JSON 拖入 ComfyUI，然后按本机文件名修改模型下拉框并上传参考图。示例中的 `moss/` 子目录和文件名是本次验证使用的名称，不是硬性要求。

多图示例中的 `example_reference.png` 是第二张图片的占位名称，请自行上传或选择。加载工作流时若提示缺少媒体输入，先完成这一步。

1. 官方 **Load Diffusion Model / Load Model Patch / Load VAE** 接入 Apply MOSS Edit；**CLIP Text Encode** 的正负条件也接入 Apply。
2. 原始 **Load Image** 直接接 Apply 的 `image`。`reference_max_edge=512` 沿用发布者默认：只缩小参考图长边，不裁剪，宽高对齐 16；编码一次后同步写入正负条件，替换其已有的参考 latent。
3. 图片另接 **缩放图像（像素）→ Get Image Size → Empty Flux 2 Latent**。这一分支仅确定输出宽高：`megapixels=1.0`、`resolution_steps=16`，保持原图宽高比。它不会把 1MP 图片送入参考编码。
4. Apply 输出的模型经过可见的 **跳过层引导（DiT 简化）** 接 KSampler；Apply 的正负输出直接接 KSampler。最后 **VAE Decode → Save Image**。

起点设置：参考长边上限 `512`，输出约 `1MP`，`strength 1`、`Euler / normal`、`30 steps`、`CFG 4`、`denoise 1`。1MP 在原生缩放节点中等于 1024×1024 个像素，非方图对齐 16 后允许少量像素数偏差。要求的是**像素宽高为 16 的倍数**，并非 latent 网格还必须为 16 的倍数。修改输出尺寸时保留这一对齐要求。

**多图入口已实际接入推理**：`image` 显示为“待编辑图像”，必接并放在序列首位；`reference_images` 是可选的原生动态 IMAGE 输入组，连接后增加端口。附加图按 `reference_0、reference_1、reference_2…` 数值顺序排列，每张图独立缩放和 VAE 编码，正负条件使用同一有序参考序列。附加端口全部不接时，现有单图工作流保持有效。

当前 LoRA 的多图效果仍需评估；这不是无效占位接口，后续新 LoRA 若继续采用同一权重格式和编辑契约，可直接替换补丁文件。前端“原图/参考图”是操作和顺序约定，底层不会额外赋予图像角色标签，实际理解取决于训练与编辑指令。模型契约变化仍需更新代码，不能承诺任意未来 LoRA 都兼容。

当前 ComfyUI Autogrow 最多支持 100 个附加端口，这是框架限制，不是模型能力或推荐图数；图数增加会增加显存和计算量。一个端口接 IMAGE 批次时，批次内图片按顺序展开为多张参考，**不是分别执行多次编辑**。通常每个口接一张 Load Image 最直观。

`reference_max_edge` 可调高，但更高分辨率能执行并不代表编辑质量已充分验证，不把 1MP 参考图作为默认推荐。输出 1MP 与参考图 1MP 是两个独立设置。

参考与输出尺度不同可能影响构图保真。本机单张图的同种子对照中，512 参考配 1MP 输出出现构图变化，1MP 参考配 1MP 输出更接近原图；这不足以作为通用质量结论。遇到构图变化时可先用 512 参考 / 0.25MP 输出对照，再测试更高参考上限。调高上限不会主动放大尺寸较小的原图。

## shift 与 SLG

支持原生 MOSS 的 ComfyUI 会在识别底模后自动配置 **inference_shift=5.0、multiplier=1.0**，标准工作流不重复添加采样配置节点。这是按模型类型选择的默认值，不会随分辨率自动估算。仅当需要自定义 shift 时，在 SLG 前连接 **ModelSamplingAuraFlow**；不要直接替换成 ModelSamplingSD3。参见 [官方原生接口说明](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)。

SLG 没有藏在 Apply 内部。标准图中显式连接 **SkipLayerGuidanceDiTSimple**：`single_layers=22`（从零计数）、`double_layers` 留空、`start_percent=0`、`end_percent=1`。这是 [MOSS 底模的官方建议](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)，并非 Edit 补丁的强制运行条件或质量保证。清空 `single_layers` 或右键旁路即可关闭，便于同种子对照。

显存不足可降低**输出分支**像素数到 `0.25MP`；TE 可在 Load CLIP 选择 `cpu`。原生 ComfyUI 负责卸载。采样起点保持空 FLUX.2 latent。原生调度没有旧 Diffusers 实现的千步网格取整，不承诺旧实现同种子逐像素一致。

## 实现与兼容范围

Edit 权重通过原生 `ModelPatcher.add_patches` 应用到模型副本，不需要预先合并或改写底模文件。合并 Q/K/V 的模型沿用官方 LoRA 切片映射。参考图保持 t=0、仅关注参考图，文本和目标图保持原 MOSS 联合注意力；底层复用原生 MOSS block、attention 后端、预取和显存管理。

官方 ModelPatchLoader 当前没有第三方格式注册 API，因此插件在启动时扩展**已注册的官方加载器**，仅接管内含 MOSS Edit 元数据的文件，其他格式继续调用原实现。没有改写 ComfyUI 核心文件。此接入点需随官方加载器接口变化维护。

这一版不保留旧 `MOSS_EDIT_MODEL` 工作流接口，请导入上述新工作流。旧模型目录可保留，原文件不会被修改。当前原生实现每次前向重新计算参考分支，尚未迁移旧实现的参考 KV 缓存。
