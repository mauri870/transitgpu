"""Checks on the source of the package: no name defined twice at the top level of a module, no unused function in
tls/window.py, and every Triton launch inside a block that makes the device of its tensors current."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "src" / "transitgpu"


def test_no_module_of_the_package_defines_a_function_or_class_twice_at_its_top_level():
    twice = []
    for path in sorted(SOURCE.rglob("*.py")):
        seen = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in seen:
                    twice.append(f"{path.relative_to(SOURCE)}:{node.lineno} {node.name}")
                seen.add(node.name)
    assert not twice, twice


def test_every_function_of_tls_window_is_used_in_the_package_or_its_tests():
    text = "\n".join(path.read_text() for path in [*SOURCE.rglob("*.py"), *(SOURCE.parent.parent / "tests").glob("*.py")])
    unused = [node.name for node in ast.parse((SOURCE / "tls" / "window.py").read_text()).body if isinstance(node, ast.FunctionDef) and text.count(node.name) == text.count(f"def {node.name}(")]
    assert not unused, unused


def launch_sites(paths):
    """Every call of a subscripted name, `NAME[grid](...)` under any name (a jit function, a variable that holds one, `kernel, triton = found`), as (file:line, inside), where inside says whether the call is in a
    `with on_device(...)` block. Nothing else of the package is called through a subscript, so each of these is a launch."""
    sites = []

    def visit(node, inside, path):
        if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Call) and getattr(item.context_expr.func, "id", getattr(item.context_expr.func, "attr", None)) == "on_device" for item in node.items):
            inside = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            sites.append((f"{Path(path).name}:{node.lineno}", inside))
        for child in ast.iter_child_nodes(node):
            visit(child, inside, path)

    for path in paths:
        visit(ast.parse(Path(path).read_text()), False, path)
    return sites


def test_every_triton_launch_of_the_package_is_inside_a_block_that_makes_the_device_of_its_tensors_current():
    """Triton launches a kernel on the current device of the process, cuda:0 unless changed, and not on the device of
    its tensors. A launch is a call of any subscripted name (transitgpu.core launches through a variable `kernel`), and
    each must be inside `with on_device(...)`."""
    sites = launch_sites(sorted(SOURCE.rglob("*.py")))
    assert {location.split(":")[0] for location, _ in sites} >= {"core.py", "binned.py", "fold.py", "prefix.py", "screen.py"}, f"the launches found: {sites}"
    bad = [location for location, inside in sites if not inside]
    assert not bad, bad


def test_the_launch_lint_flags_a_launch_outside_the_block_under_any_name_and_not_one_inside_it(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import triton\n\n@triton.jit\ndef _anything(x):\n    pass\n\nalias = _anything\n\ndef run(x, device, found):\n    _anything[(1,)](x)\n    alias[(1,)](x)\n    with on_device(device):\n        alias[(1,)](x)\n    with other():\n        _anything[(2,)](x)\n    kernel, triton = found\n    kernel[(3,)](x)\n")
    assert launch_sites([probe]) == [("probe.py:10", False), ("probe.py:11", False), ("probe.py:13", True), ("probe.py:15", False), ("probe.py:17", False)]
