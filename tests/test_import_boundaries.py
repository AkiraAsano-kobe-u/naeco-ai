"""Static import guardrails; runnable with unittest alone (no torch/pytest).

Only train/net.py and train/ppo.py need direct torch imports. Web agents load
naeco.train lazily via importlib for normal/strong play; internal naeco imports
are allowed and dynamic import strings are deliberately outside this check.
No extra torch exception is needed, including for imports inside functions or
TYPE_CHECKING blocks. This checks direct imports, not transitive dependencies.
"""
import ast
from importlib.machinery import PathFinder
from pathlib import Path
import sys
import sysconfig
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TORCH_ALLOWED = {"naeco/train/net.py", "naeco/train/ppo.py"}
STDLIB_ONLY = ("naeco/ai/search/", "naeco/ui/web/")


def _is_stdlib(name):
    # Python 3.9 has no sys.stdlib_module_names. Search *only* interpreter
    # stdlib/extension directories, never sys.path (which includes site-packages
    # and the working directory). Finding a spec does not execute the module.
    if name in sys.builtin_module_names:
        return True
    paths = {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")}
    # Apple's relocated Python can leave DESTSHARED pointing at a nonexistent
    # Xcode directory; extensions live under the actual stdlib in that case.
    paths.update(str(Path(p) / "lib-dynload") for p in tuple(paths) if p)
    paths.add(sysconfig.get_config_var("DESTSHARED"))
    return PathFinder.find_spec(name, sorted(p for p in paths if p)) is not None


def _import_violations(path, relative_name):
    violations = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            name = module.split(".")[0]
            reason = None
            if name == "torch" and relative_name not in TORCH_ALLOWED:
                reason = "torch import outside train/net.py and train/ppo.py"
            elif relative_name.startswith(STDLIB_ONLY) and name != "naeco" \
                    and not _is_stdlib(name):
                reason = "non-stdlib/non-naeco import"
            if reason:
                violations.append("%s:%d: %s (%s)" % (
                    relative_name, node.lineno, module, reason))
    return violations


class ImportBoundaryTests(unittest.TestCase):
    def test_naeco_import_boundaries(self):
        paths = sorted((ROOT / "naeco").rglob("*.py"))
        self.assertTrue(paths, "no naeco Python sources found")
        violations = []
        for path in paths:
            violations.extend(_import_violations(
                path, path.relative_to(ROOT).as_posix()))
        self.assertFalse(violations, "\n" + "\n".join(violations))

    def test_stdlib_resolution_excludes_third_party_and_local_modules(self):
        for name in ("sys", "math", "json", "http", "importlib", "__future__"):
            with self.subTest(name=name):
                self.assertTrue(_is_stdlib(name))
        for name in ("pytest", "torch", "naeco", "not_a_real_dependency"):
            with self.subTest(name=name):
                self.assertFalse(_is_stdlib(name))

    def test_detector_rejects_forbidden_imports(self):
        cases = [
            ("naeco/engine/probe.py", "import os, torch as t\n"),
            ("naeco/train/agent.py", "from torch.nn import Linear\n"),
            ("naeco/ui/web/probe.py", "def lazy():\n    import torch\n"),
            ("naeco/ai/search/probe.py", "if False:\n    import numpy\n"),
            ("naeco/ui/web/probe.py", "from pytest import raises\n"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.py"
            for relative_name, source in cases:
                with self.subTest(path=relative_name, source=source):
                    path.write_text(source, encoding="utf-8")
                    violations = _import_violations(path, relative_name)
                    self.assertEqual(len(violations), 1)
                    self.assertIn(relative_name + ":", violations[0])

    def test_detector_allows_stdlib_internal_and_training_imports(self):
        cases = [
            ("naeco/train/net.py", "import torch.nn as nn\n"),
            ("naeco/train/ppo.py", "from torch import Tensor\n"),
            ("naeco/ui/web/probe.py", "import http.server, math\n"
             "from naeco.train import net\nfrom ...engine import state\n"),
            ("naeco/ai/search/probe.py", "import collections, time\n"
             "from naeco.engine.state import State\n"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.py"
            for relative_name, source in cases:
                with self.subTest(path=relative_name, source=source):
                    path.write_text(source, encoding="utf-8")
                    self.assertEqual(_import_violations(path, relative_name), [])


if __name__ == "__main__":
    unittest.main()
