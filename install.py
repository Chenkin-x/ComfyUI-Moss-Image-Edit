"""Read-only installation preflight, also used by ComfyUI-Manager."""
import argparse
import ast
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
except ImportError:
    # pip already ships this parser; checking an environment must not install it.
    try:
        from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
        from pip._vendor.packaging.specifiers import SpecifierSet
        from pip._vendor.packaging.utils import canonicalize_name
    except ImportError:
        print("[MOSS Edit] 检查未完成：packaging 和 pip 的依赖解析器均不可用。请核对所选 Python。\n"
              "Check incomplete: neither packaging nor pip's parser is available. Verify your interpreter.\n"
              "No packages changed.", file=sys.stderr)
        raise SystemExit(3)


NATIVE_APIS = {
    "comfy/model_base.py": ("MOSSImage",),
    "comfy/ldm/moss_image/model.py": ("MOSSImage", "modulated_rms_norm"),
    "comfy/text_encoders/moss_image.py": ("MOSSImageTokenizer", "MOSSImageTEModel"),
    "comfy/model_prefetch.py": (
        "make_prefetch_queue", "prefetch_queue_pop", "malloc_graph_begin", "malloc_graph_end"),
    "comfy_extras/nodes_model_patch.py": ("ModelPatchLoader",),
    "comfy_extras/nodes_flux.py": ("EmptyFlux2LatentImage",),
    "comfy_extras/nodes_slg.py": ("SkipLayerGuidanceDiTSimple",),
    "comfy_extras/nodes_post_processing.py": ("ImageScaleToTotalPixels",),
    "comfy_extras/nodes_images.py": ("GetImageSize",),
}


def core_version(root):
    path = root / "comfyui_version.py"
    if path.exists():
        for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
            ):
                return ast.literal_eval(node.value)
    return "unknown (capabilities will still be checked)"


def native_api_errors(root):
    errors = []
    for filename, names in NATIVE_APIS.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"缺少原生接口 / Missing native module: {filename}")
            continue
        declared = {node.name for node in ast.parse(path.read_text(encoding="utf-8-sig")).body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for name in names:
            if name not in declared:
                errors.append(f"原生接口不匹配 / Missing native API: {filename} :: {name}")
    return errors


def installed_packages():
    # Match importlib.metadata's first-distribution precedence on sys.path.
    result = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result.setdefault(canonicalize_name(name), {
                "version": dist.version, "requires": dist.requires or [],
            })
    return result


def requirement_problem(req, packages):
    if req.marker and not req.marker.evaluate({"extra": ""}):
        return None
    installed = packages.get(canonicalize_name(req.name))
    if installed is None:
        return f"{req}: 未安装 / not installed"
    if req.url:
        return f"{req.name}: URL requirement needs manual provenance verification"
    if req.specifier and not req.specifier.contains(installed["version"], prereleases=True):
        return f"{req}: 已安装 / installed {installed['version']}"
    return None


def core_requirements(root, packages):
    errors, warnings = [], []
    optional = False
    for number, line in enumerate((root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines(), 1):
        if "non essential dependencies" in line.lower():
            optional = True
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            errors.append(f"requirements.txt:{number}: 无法检查 / unsupported requirement: {line}")
            continue
        problem = requirement_problem(req, packages)
        if problem:
            (warnings if optional else errors).append("ComfyUI requires " + problem)
    return errors, warnings


def existing_conflicts(packages):
    conflicts = []
    for name, package in sorted(packages.items()):
        for raw in package["requires"]:
            try:
                problem = requirement_problem(Requirement(raw), packages)
            except InvalidRequirement:
                problem = f"无法解析依赖声明 / invalid requirement: {raw}"
            if problem:
                conflicts.append(f"{name} {package['version']} requires {problem}")
    return conflicts


def probe(root):
    # Separate CPU-only process: no other custom nodes, model loading or pip.
    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.model_base
    import comfy.model_patcher
    import comfy.model_prefetch
    import comfy.supported_models
    import comfy.text_encoders.moss_image
    import comfy.ldm.moss_image.model
    import nodes
    from comfy_extras.nodes_model_patch import ModelPatchLoader
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
    from comfy_extras.nodes_images import GetImageSize
    from comfy_api.latest import io

    io.Autogrow.TemplatePrefix(io.Image.Input("reference", optional=True), prefix="reference_", min=0, max=100)

    assert hasattr(comfy.model_patcher.ModelPatcher, "add_object_patch"), "ModelPatcher.add_object_patch missing"
    assert hasattr(comfy.model_patcher, "CoreModelPatcher"), "CoreModelPatcher missing"
    assert hasattr(comfy.ldm.moss_image.model.MOSSImage, "_forward"), "native MOSS _forward missing"
    assert "moss_image" in nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0], "CLIPLoader moss_image type missing"
    settings = comfy.supported_models.MOSSImage.sampling_settings
    assert settings.get("shift") == 5.0 and settings.get("multiplier") == 1.0, f"MOSS sampling defaults changed: {settings}"
    for cls in (EmptyFlux2LatentImage, SkipLayerGuidanceDiTSimple, ImageScaleToTotalPixels, GetImageSize):
        cls.define_schema()
    assert callable(ModelPatchLoader.load_model_patch), "ModelPatchLoader API missing"
    print("Native MOSS / TE / model patch / workflow imports passed (CPU, no weights loaded).")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MOSS Edit 环境检查 / read-only environment check (no pip installs)")
    parser.add_argument("--comfyui", type=Path, help="ComfyUI root, if this plugin is outside custom_nodes")
    parser.add_argument("--report", type=Path, help="Optional JSON diagnostic report destination")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = (args.comfyui or Path(__file__).resolve().parents[2]).resolve()
    if args.probe:
        probe(root)
        return 0
    print("[MOSS Edit] 安装前环境检查 / Installation preflight", flush=True)
    print(f"Python: {sys.executable}\nPython version: {sys.version.split()[0]}\nComfyUI: {root}", flush=True)
    print("只检查，不修改现有环境、不下载模型 / Read-only; no package changes or model downloads.", flush=True)
    errors, warnings = [], []
    report = {"python": sys.executable, "python_version": sys.version.split()[0], "comfyui": str(root),
              "errors": errors, "warnings": warnings, "environment_modified": False}
    if not (root / "comfy").is_dir() or not (root / "requirements.txt").is_file():
        errors.append("未找到 ComfyUI 核心；用 --comfyui 指定根目录 / ComfyUI root not found; pass --comfyui.")
    else:
        report["comfyui_version"] = core_version(root)
        print(f"ComfyUI version: {report['comfyui_version']} (tested baseline: 0.42.3)", flush=True)
        project = root / "pyproject.toml"
        if project.exists():
            python_req = re.search(r'^requires-python\s*=\s*"([^"]+)"', project.read_text(encoding="utf-8-sig"), re.MULTILINE)
            if python_req and not SpecifierSet(python_req[1]).contains(sys.version.split()[0], prereleases=True):
                errors.append(f"ComfyUI requires Python {python_req[1]}; installed {sys.version.split()[0]}")
        api_errors = native_api_errors(root)
        errors.extend(api_errors)
        packages = installed_packages()
        report["installed_versions"] = {name: item["version"] for name, item in sorted(packages.items())}
        missing, optional = core_requirements(root, packages)
        errors.extend(missing)
        warnings.extend(optional)
        report["existing_dependency_conflicts"] = existing_conflicts(packages)
        # Even on metadata mismatch, distinguish actual import failures from declared requirements.
        if not api_errors:
            print("检查原生接口导入（CPU，不加载模型） / Checking native imports...", flush=True)
            try:
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--comfyui", str(root), "--probe"],
                    cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
                report["import_probe"] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                if result.returncode:
                    errors.append("原生接口导入失败 / Native imports failed:\n" + (result.stderr or result.stdout)[-6000:])
            except subprocess.TimeoutExpired:
                errors.append("原生接口检查超过 90 秒 / Native import check timed out; inspect the Python environment manually.")
    for message in errors:
        print("[FAIL] " + message, flush=True)
    for message in warnings:
        print("[WARN] " + message, flush=True)
    conflicts = report.get("existing_dependency_conflicts", [])
    if conflicts:
        print("[WARN] 以下是当前环境已有的依赖声明冲突，不自动修复，也不单独作为 MOSS 失败依据：\n"
              "Existing metadata conflicts (advisory; no automatic repairs):", flush=True)
        for conflict in conflicts:
            print("  " + conflict, flush=True)
    report["status"] = "insufficient" if errors else "passed_with_warnings" if warnings or conflicts else "passed"
    if errors:
        print("[MOSS Edit] 依赖或核心接口不充分，节点可能无法正常使用。原环境未修改。\n"
              "请用启动 ComfyUI 的同一个 Python 核对上方冲突；核心/API 问题通过原启动器处理。\n"
              "Environment insufficient; resolve the listed issues using your launcher/environment owner, then rerun.\n"
              "不要直接全量 pip upgrade；请参阅 README 的安装诊断。", flush=True)
    else:
        print("[MOSS Edit] 环境预检通过。重启 ComfyUI 后导入标准工作流。\n"
              "Preflight passed. Restart ComfyUI and import the standard workflow.\n"
              "此检查不验证模型文件、GPU 驱动、显存或出图效果 / Weights, GPU and inference quality are not tested.", flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {args.report.resolve()}", flush=True)
    return 2 if errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError) as error:
        print(f"[MOSS Edit] 检查未完成 / Check incomplete: {error}. No packages changed.", file=sys.stderr)
        sys.exit(3)
