"""ComfyUI custom-node entry point."""
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as error:
    raise ImportError(
        "[MOSS Edit] 原生接口或依赖导入失败，节点可能无法正常使用。"
        "请使用启动 ComfyUI 的同一个 Python 运行本插件的 install.py 查看具体缺项；"
        "不会自动修改现有环境。 / Native import failed; run this plugin's install.py "
        f"with ComfyUI's Python for diagnostics. Original error: {error}"
    ) from error

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
