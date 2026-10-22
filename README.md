<p align="center">
  <strong>简体中文</strong> &nbsp; · &nbsp; <a href="README.en.md">English</a>
</p>

<p align="center">
  <img src="assets/moss-image-edit-banner-cases.png" alt="MOSS Image Edit — 用文字编辑图像" width="1280">
</p>

<h1 align="center">MOSS Image Edit for ComfyUI</h1>

<p align="center">
  <strong>上传参考图，写下变化，点击生成。</strong><br>
  在 ComfyUI 中使用 MOSS Image Edit <strong>图像编辑 LoRA</strong>，修改表情、穿搭、物体与场景，支持单图编辑和最多 4 张参考图。<br>
  需要同时加载 MOSS-Image Base 或 SFT 底模与本项目的 LoRA 权重。
</p>

<p align="center">
  <a href="#安装">安装</a> &nbsp; · &nbsp;
  <a href="#下载模型">模型下载</a> &nbsp; · &nbsp;
  <a href="#运行工作流">参考工作流</a> &nbsp; · &nbsp;
  <a href="https://github.com/Chenkin-x/Moss-Image-Edit#编辑展示">编辑效果</a>
</p>

## 安装

旧版用户请先将 `custom_nodes/Moss-Image-Edit` 移出 `custom_nodes`，避免重复注册节点；模型目录保持原位。

在 **ComfyUI 使用的 Python 环境**中执行：

```bash
cd /path/to/ComfyUI
git clone https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit.git custom_nodes/ComfyUI-Moss-Image-Edit
python -m pip install -r custom_nodes/ComfyUI-Moss-Image-Edit/requirements.txt
```

重启 ComfyUI，刷新页面。节点菜单的 `MOSS Image Edit` 分类下会出现模型加载与图像编辑节点。Windows 便携版请使用它自带的 `python_embeded/python.exe` 安装依赖，安装方式也可参照 [ComfyUI 官方说明](https://docs.comfy.org/installation/install_custom_node)。

## 下载模型

底模支持 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) 和 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)，**任选其一**，再下载本项目的 [编辑 LoRA 权重](https://huggingface.co/chinoll/Moss-Image-Edit)。以下以 SFT 为例，在 ComfyUI 根目录执行：

```bash
hf download OpenMOSS-Team/MOSS-Image-SFT \
  --local-dir models/moss_image/MOSS-Image-SFT

hf download chinoll/Moss-Image-Edit \
  adapter_config.json adapter_model.safetensors \
  --local-dir models/moss_image/Moss-Image-Edit
```

选择 Base 时，底模下载命令改为：

```bash
hf download OpenMOSS-Team/MOSS-Image-Base \
  --local-dir models/moss_image/MOSS-Image-Base
```

目录结构如下（以 SFT 为例）：

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

模型加载节点的 `snapshot_dir` 指定完整底模目录，`adapter_dir` 指定 LoRA 权重目录，默认对应上面两个目录；相对路径以 `ComfyUI/models` 为基准，也可以填写绝对路径。

使用 Base 时，将模型加载节点的 `snapshot_dir` 设置为 `moss_image/MOSS-Image-Base`。

## 运行工作流

| 场景 | 拖入 ComfyUI 的工作流 |
|:---|:---|
| 编辑一张图 | [moss_edit_single_reference.json](workflows/moss_edit_single_reference.json) |
| 两张图共同引导编辑 | [moss_edit_multi_reference.json](workflows/moss_edit_multi_reference.json) |

1. 下载工作流 JSON，拖入 ComfyUI 页面。
2. 在每个 `Load Image` 节点上传自己的参考图。
3. 在 `MOSS Image Edit` 中填写 `prompt`，例如 `Change the background to a sunny beach, preserving the person.`。`negative_prompt` 可以留空。
4. 设置输出 `width`、`height`，点击 **Run**。结果在 `Save Image` 节点预览，并保存到 `ComfyUI/output/MOSS-Image-Edit/`。

多图工作流已连接两张参考图。继续添加 `Load Image` 并接入 `reference_3`、`reference_4`，即可使用三张或四张图；在指令中用 image 1、image 2 等说明各图的作用，例如 `Dress the person in image 1 in the outfit from image 2.`。每个输入接一张图，各张图可以有不同的宽高比。

## 参数

| 参数 | 工作流默认值 | 用途 |
|:---|:---|:---|
| `width` / `height` | `512` / `512` | 输出尺寸，均为 16 的倍数。 |
| `reference_max_edge` | `512` | 参考图长边上限；缩小后宽高对齐到 16，不裁剪。 |
| `steps` | `30` | 去噪步数。 |
| `cfg` | `4.0` | 文本引导强度。 |
| `seed` | `42`，`fixed` | 固定初始噪声；比较编辑参数时保持不变。 |
| `inference_shift` | `5.0` | 采样时间分布。 |

API 调用示例见 [单图 API JSON](workflows/moss_edit_single_reference.api.json) 和 [多图 API JSON](workflows/moss_edit_multi_reference.api.json)。其中图片名对应 `ComfyUI/input/` 下的文件，修改后作为 ComfyUI `/prompt` 请求中的 `prompt` 对象使用。

---

本项目代码采用 [GNU GPLv3](LICENSE)。
