"""Preflight failure cases without changing any installed distribution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("moss_install", Path(__file__).parents[1] / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="moss-install-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename, names in installer.NATIVE_APIS.items():
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(f"class {name}: pass" for name in names), encoding="utf-8")
        (self.root / "requirements.txt").write_text("example>=2\n", encoding="utf-8")
        (self.root / "comfyui_version.py").write_text('__version__ = "0.42.3"\n', encoding="utf-8")
        self.packages = {"example": {"version": "2.5", "requires": []}}

    def run_check(self, result=None, side_effect=None):
        with patch.object(installer, "installed_packages", return_value=self.packages), \
             patch.object(installer.subprocess, "run", return_value=result or subprocess.CompletedProcess([], 0, "OK", ""),
                          side_effect=side_effect) as child, contextlib.redirect_stdout(io.StringIO()) as output:
            code = installer.main(["--comfyui", str(self.root), "--report", str(self.root / "report.json")])
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["environment_modified"])
        return code, report, output.getvalue(), child

    def test_success_uses_same_python_and_cpu_probe_not_pip(self):
        code, report, _, child = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        command = child.call_args.args[0]
        self.assertEqual(command[0], installer.sys.executable)
        self.assertIn("--probe", command)
        self.assertNotIn("pip", command)

    def test_missing_package_is_explicit_and_fails(self):
        self.packages.clear()
        code, _, output, _ = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("example>=2", output)
        self.assertIn("not installed", output)

    def test_insufficient_version_shows_requirement_and_current(self):
        self.packages["example"]["version"] = "1.0"
        code, _, output, _ = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("example>=2", output)
        self.assertIn("installed 1.0", output)

    def test_old_core_stops_before_imports(self):
        (self.root / "comfy/ldm/moss_image/model.py").unlink()
        code, _, output, child = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("Missing native module", output)
        child.assert_not_called()

    def test_existing_unrelated_conflicts_are_advisory(self):
        self.packages["old-plugin"] = {"version": "1", "requires": ["example<2"]}
        code, report, output, _ = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertIn("old-plugin 1 requires example<2", output)

    def test_optional_core_requirement_and_extras_do_not_block(self):
        (self.root / "requirements.txt").write_text(
            'example>=2\n#non essential dependencies:\noptional-lib>=1\n', encoding="utf-8")
        self.packages["example"]["requires"] = ['extra-only; extra == "training"', 'win-only; sys_platform == "fiction"']
        code, report, _, _ = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertEqual(report["existing_dependency_conflicts"], [])

    def test_native_import_failure_preserves_actionable_exception(self):
        code, _, output, _ = self.run_check(subprocess.CompletedProcess([], 1, "", "ImportError: broken DLL example"))
        self.assertEqual(code, 2)
        self.assertIn("ImportError: broken DLL example", output)

    def test_probe_timeout_does_not_claim_success(self):
        code, _, output, _ = self.run_check(side_effect=subprocess.TimeoutExpired("python", 90))
        self.assertEqual(code, 2)
        self.assertIn("timed out", output)

    def test_wrong_core_path_does_not_probe(self):
        (self.root / "requirements.txt").unlink()
        code, _, output, child = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("--comfyui", output)
        child.assert_not_called()

    def test_unhandled_requirement_syntax_is_reported_not_skipped(self):
        (self.root / "requirements.txt").write_text("-r another-file.txt\n", encoding="utf-8")
        code, _, output, _ = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("unsupported requirement", output)

    def test_python_requirement_is_checked_against_current_core(self):
        (self.root / "pyproject.toml").write_text('requires-python = ">=99"\n', encoding="utf-8")
        code, _, output, _ = self.run_check()
        self.assertEqual(code, 2)
        self.assertIn("ComfyUI requires Python >=99", output)


if __name__ == "__main__":
    unittest.main()
