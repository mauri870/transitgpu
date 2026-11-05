"""The accumulation of the box search in the local memory of the device (transitgpu.hip, ROCm only): it gives the integers of the Triton kernel to the bit, and without a compiler the Triton kernel is used.

Function tests: the property is that two paths give equal integers on random inputs, and the fallback is a code path with no data.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu import core, hip

on_rocm = bool(torch.version.hip) and torch.cuda.is_available()
needs_rocm = pytest.mark.skipif(not on_rocm, reason="the HIP kernel runs on ROCm only")


def inputs(points=3000, fluxes=5, seed=0, weights=None):
    generator = np.random.default_rng(seed)
    t = torch.as_tensor(np.sort(generator.uniform(0.0, 400.0, points)), dtype=torch.float64, device="cuda")
    quantized = torch.as_tensor(generator.integers(-2_000_000, 2_000_000, (fluxes, points)), dtype=torch.int64, device="cuda")
    weight = torch.ones((fluxes, points), dtype=torch.int64, device="cuda") if weights is None else weights
    return t, quantized, weight


def triton_result(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(hip, "accumulate_many", lambda *a, **k: None)
        return core.accumulate_many(*args, **kwargs)


@needs_rocm
@pytest.mark.parametrize("bins", [128, 256, 512, 1024, 2048, 4096])
def test_the_integers_equal_those_of_the_triton_kernel_for_every_bin_count_of_the_grid(bins, monkeypatch):
    t, quantized, weight = inputs()
    p = torch.as_tensor(np.linspace(3.0, 90.0, 40), dtype=torch.float64, device="cuda")
    sums, counts = core.accumulate_many(t, p, bins, quantized, weight, shared_weight=True)
    reference_sums, reference_counts = triton_result(monkeypatch, t, p, bins, quantized, weight, shared_weight=True)
    assert torch.equal(sums, reference_sums), f"{bins} bins: the sums differ in {int((sums != reference_sums).sum())} bins"
    assert torch.equal(counts, reference_counts), f"{bins} bins: the counts differ"


@needs_rocm
def test_the_kernel_of_the_device_is_the_one_that_ran_on_the_shuffles_of_a_stack(monkeypatch):
    t, quantized, weight = inputs(points=20_000, fluxes=13, seed=1)
    p = torch.as_tensor(np.linspace(1.0, 300.0, 512), dtype=torch.float64, device="cuda")
    calls = []
    original = hip.accumulate_many
    monkeypatch.setattr(hip, "accumulate_many", lambda *a, **k: calls.append(1) or original(*a, **k))
    sums, counts = core.accumulate_many(t, p, 1024, quantized, weight, shared_weight=True)
    assert calls, "the HIP kernel was not asked"
    reference_sums, reference_counts = triton_result(monkeypatch, t, p, 1024, quantized, weight, shared_weight=True)
    assert torch.equal(sums, reference_sums) and torch.equal(counts, reference_counts)


@needs_rocm
def test_equal_weights_that_are_not_ones_give_the_integers_of_the_triton_kernel(monkeypatch):
    t, quantized, _ = inputs(seed=2)
    row = torch.as_tensor(np.random.default_rng(3).integers(0, 40, quantized.shape[1]), dtype=torch.int64, device="cuda")
    weight = row.expand(quantized.shape[0], -1).contiguous()
    p = torch.as_tensor(np.linspace(5.0, 60.0, 17), dtype=torch.float64, device="cuda")
    sums, counts = core.accumulate_many(t, p, 512, quantized, weight, shared_weight=True)
    reference_sums, reference_counts = triton_result(monkeypatch, t, p, 512, quantized, weight, shared_weight=True)
    assert torch.equal(sums, reference_sums) and torch.equal(counts, reference_counts)


@needs_rocm
def test_weights_that_differ_between_fluxes_go_to_the_triton_kernel(monkeypatch):
    t, quantized, _ = inputs(seed=4)
    weight = torch.as_tensor(np.random.default_rng(5).integers(0, 3, quantized.shape), dtype=torch.int64, device="cuda")
    p = torch.as_tensor(np.linspace(5.0, 60.0, 9), dtype=torch.float64, device="cuda")
    assert hip.accumulate_many(t, p, 512, quantized, weight, shared_weight=False) is None
    sums, counts = core.accumulate_many(t, p, 512, quantized, weight)
    reference_sums, reference_counts = triton_result(monkeypatch, t, p, 512, quantized, weight)
    assert torch.equal(sums, reference_sums) and torch.equal(counts, reference_counts)


@needs_rocm
def test_weights_too_large_for_the_counts_of_a_block_go_to_the_triton_kernel():
    t, quantized, _ = inputs(seed=6)
    weight = torch.full(quantized.shape, 2**40, dtype=torch.int64, device="cuda")
    p = torch.as_tensor(np.linspace(5.0, 60.0, 9), dtype=torch.float64, device="cuda")
    assert hip.accumulate_many(t, p, 512, quantized, weight, shared_weight=True) is None


@needs_rocm
@pytest.mark.parametrize("points, rows", [(1, 1), (2, 3), (4096, 1), (4097, 2), (20_000, 512)])
def test_one_point_an_empty_bin_and_the_largest_batch_give_the_integers_of_the_triton_kernel(points, rows, monkeypatch):
    t, quantized, weight = inputs(points=points, fluxes=3, seed=7)
    p = torch.as_tensor(np.linspace(200.0, 399.0, rows), dtype=torch.float64, device="cuda")
    sums, counts = core.accumulate_many(t, p, 4096, quantized, weight, shared_weight=True)
    reference_sums, reference_counts = triton_result(monkeypatch, t, p, 4096, quantized, weight, shared_weight=True)
    assert torch.equal(sums, reference_sums) and torch.equal(counts, reference_counts)
    assert int((counts[0] == 0).sum()) > 0 or points > 4096 * rows, "the case has no empty bin"


def test_a_failed_compile_warns_once_and_leaves_the_triton_kernel_to_the_caller(monkeypatch):
    from torch.utils import cpp_extension

    def broken(*args, **kwargs):
        raise RuntimeError("hipcc: command not found")

    monkeypatch.setattr(hip, "_module", None)
    monkeypatch.setattr(hip, "_failed", False)
    monkeypatch.setattr(hip, "available", lambda: True)
    monkeypatch.setattr(cpp_extension, "load", broken)
    t = torch.zeros(4, dtype=torch.float64)
    quantized = torch.zeros((3, 4), dtype=torch.int64)
    with pytest.warns(UserWarning, match="hipcc: command not found"):
        assert hip.accumulate_many(t, torch.ones(1, dtype=torch.float64), 128, quantized, torch.ones_like(quantized), shared_weight=True) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hip.accumulate_many(t, torch.ones(1, dtype=torch.float64), 128, quantized, torch.ones_like(quantized), shared_weight=True) is None


def test_off_rocm_the_kernel_is_not_asked(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert hip.available() is False


# The build of the extension (W43): the architecture is fixed so that the build is cached, and a lock left by a killed build is waited on for a bounded time.


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(hip, "_module", None)
    monkeypatch.setattr(hip, "_failed", False)
    monkeypatch.setattr(hip, "_built_for", hip._built_for)  # a build in tmp_path sets it, and a later test would see the architectures of that build
    monkeypatch.setattr(hip, "_build_directory", lambda: tmp_path)
    return tmp_path


def test_a_lock_left_by_a_killed_build_makes_the_load_warn_and_leave_the_triton_kernel_to_the_caller_within_the_bound(fresh, monkeypatch):
    import time

    (fresh / "lock").write_text("")
    monkeypatch.setattr(hip, "LOCK_WAIT_SECONDS", 0.4)
    started = time.perf_counter()
    with pytest.warns(UserWarning, match="lock"):
        assert hip._load() is None
    assert time.perf_counter() - started < 5.0 and hip._failed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hip._load() is None, "the second call does not wait and does not warn again"


def test_a_lock_released_during_the_wait_lets_the_build_go_on_with_the_architecture_of_the_device_set_for_the_build_only(fresh, monkeypatch):
    import os
    import threading

    from torch.utils import cpp_extension

    (fresh / "lock").write_text("")
    threading.Timer(0.3, lambda: (fresh / "lock").unlink()).start()
    seen = {}

    def fake_load(**options):
        seen.update(options, arch=os.environ.get("PYTORCH_ROCM_ARCH"))
        return "module"

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(hip, "_architecture", lambda: "gfx1100")
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    assert hip._load() == "module"
    assert seen["arch"] == "gfx1100" and seen["build_directory"] == str(fresh) and seen["name"] == "transitgpu_accumulate"
    assert "PYTORCH_ROCM_ARCH" not in os.environ, "the variable is set for the build and removed after it"


def test_an_architecture_that_the_user_set_is_kept(fresh, monkeypatch):
    import os

    from torch.utils import cpp_extension

    seen = {}
    monkeypatch.setattr(cpp_extension, "load", lambda **options: seen.update(arch=os.environ.get("PYTORCH_ROCM_ARCH")) or "module")
    monkeypatch.setattr(hip, "_architecture", lambda: "gfx1100")
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx90a")
    hip._load()
    assert seen["arch"] == "gfx90a" and os.environ["PYTORCH_ROCM_ARCH"] == "gfx90a"


def test_the_architecture_is_that_of_the_device_without_its_features(monkeypatch):
    import types

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: types.SimpleNamespace(gcnArchName="gfx1100:sramecc+:xnack-"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert hip._architecture() == "gfx1100"
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: types.SimpleNamespace())
    assert hip._architecture() is None


def test_the_build_directory_is_one_of_the_environment_and_the_users_choice_of_torch_wins(monkeypatch, tmp_path):
    import sys

    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert hip._build_directory() == tmp_path / "transitgpu_extension"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "mine"))
    assert hip._build_directory() is None, "torch's own directory, which the variable names"
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "no" / "such" / "readonly"))
    monkeypatch.setattr("os.access", lambda *a, **k: False)
    assert hip._build_directory() is None, "an environment that cannot be written to falls back to torch's directory"


@needs_rocm
def test_a_second_process_loads_the_build_of_the_first_without_compiling(tmp_path):
    """The extension was compiled again in every process because torch listed the architectures of the cards in an order that changes; with the architecture fixed the second process finds the build."""
    import os
    import subprocess
    import sys

    program = "import time, transitgpu.hip as h; t = time.perf_counter(); assert h._load() is not None; print(time.perf_counter() - t)"
    environment = {**os.environ, "TORCH_EXTENSIONS_DIR": str(tmp_path)}
    environment.pop("PYTORCH_ROCM_ARCH", None)
    seconds = [float(subprocess.run([sys.executable, "-c", program], env=environment, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]) for _ in range(2)]
    assert seconds[1] < 5.0, f"the first process took {seconds[0]:.1f} s and the second {seconds[1]:.1f} s"


# The architecture of the device against the one the extension was built for (W43): a kernel built for one architecture is not launched on a device of another.


class FakeModule:
    def __init__(self):
        self.calls = 0

    def accumulate(self, *arguments):
        self.calls += 1


def call_accumulate(monkeypatch, built_for, device_arch, user_arch=None):
    fake = FakeModule()
    monkeypatch.setattr(hip, "_module", fake)
    monkeypatch.setattr(hip, "_failed", False)
    monkeypatch.setattr(hip, "_built_for", built_for)
    monkeypatch.setattr(hip, "_warned_architectures", set())
    monkeypatch.setattr(hip, "available", lambda: True)
    monkeypatch.setattr(hip, "_device_architecture", lambda device: device_arch)
    t = torch.zeros(4, dtype=torch.float64)
    quantized = torch.zeros((3, 4), dtype=torch.int64)
    found = hip.accumulate_many(t, torch.ones(2, dtype=torch.float64), 128, quantized, torch.ones_like(quantized), shared_weight=True)
    return fake, found


def test_a_device_of_another_architecture_than_the_build_leaves_the_triton_kernel_to_the_caller_and_says_so_once(monkeypatch):
    with pytest.warns(UserWarning, match="gfx1100.*gfx1030"):
        fake, found = call_accumulate(monkeypatch, {"gfx1100"}, "gfx1030")
    assert found is None and fake.calls == 0
    t = torch.zeros(4, dtype=torch.float64)
    quantized = torch.zeros((3, 4), dtype=torch.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hip.accumulate_many(t, torch.ones(2, dtype=torch.float64), 128, quantized, torch.ones_like(quantized), shared_weight=True) is None, "the second call does not warn again"


def test_a_device_of_the_architecture_of_the_build_runs_the_kernel(monkeypatch):
    fake, found = call_accumulate(monkeypatch, {"gfx1100"}, "gfx1100")
    assert fake.calls == 1 and found is not None


def test_a_build_for_several_architectures_runs_on_each_of_them(monkeypatch):
    for architecture in ("gfx1100", "gfx90a"):
        fake, found = call_accumulate(monkeypatch, {"gfx1100", "gfx90a"}, architecture)
        assert fake.calls == 1


def test_an_unknown_architecture_of_the_build_or_of_the_device_is_not_checked(monkeypatch):
    for built_for, device_arch in ((None, "gfx1030"), ({"gfx1100"}, None)):
        fake, found = call_accumulate(monkeypatch, built_for, device_arch)
        assert fake.calls == 1


def test_the_architectures_of_a_build_are_the_chosen_one_or_the_list_the_user_set():
    assert hip._architectures("gfx1100", None) == {"gfx1100"}
    assert hip._architectures(None, "gfx1100;gfx90a") == {"gfx1100", "gfx90a"}
    assert hip._architectures(None, "gfx1100, gfx90a:xnack+") == {"gfx1100", "gfx90a"}
    assert hip._architectures(None, None) is None


def test_on_torch_without_a_device_the_architecture_is_none_and_not_an_error(monkeypatch):
    """CPU torch (the CI command) raises an AssertionError, not a RuntimeError, from torch.cuda.get_device_properties; the guard must read it as 'no architecture'."""
    def cpu_torch(*args, **kwargs):
        raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(torch.cuda, "get_device_properties", cpu_torch)
    monkeypatch.setattr(torch.cuda, "current_device", cpu_torch)
    assert hip._architecture() is None
    assert hip._device_architecture(torch.device("cpu")) is None


@needs_rocm
@pytest.mark.parametrize("bins", [128, 512, 1024, 2048, 4096])
def test_a_single_flux_runs_on_the_hip_kernel_and_gives_the_integers_of_the_triton_kernel(bins, monkeypatch):
    t, quantized, weight = inputs(fluxes=1, seed=11)
    p = torch.as_tensor(np.linspace(3.0, 90.0, 40), dtype=torch.float64, device="cuda")
    asked = []
    original = hip.accumulate_many
    monkeypatch.setattr(hip, "accumulate_many", lambda *a, **k: asked.append(original(*a, **k)) or asked[-1])
    sums, counts = core.accumulate_many(t, p, bins, quantized, weight, shared_weight=True)
    assert asked and asked[0] is not None, "the HIP kernel did not take a single flux"
    reference_sums, reference_counts = triton_result(monkeypatch, t, p, bins, quantized, weight, shared_weight=True)
    assert sums.shape == (1, 40, bins) and torch.equal(sums, reference_sums), f"{bins} bins: the sums differ in {int((sums != reference_sums).sum())} bins"
    assert torch.equal(counts, reference_counts), f"{bins} bins: the counts differ"


@needs_rocm
@pytest.mark.parametrize("points, rows", [(1, 1), (2, 3), (4097, 2), (153_046, 512)])
def test_a_single_flux_of_one_point_the_largest_bin_count_and_a_full_batch_gives_the_integers_of_the_triton_kernel(points, rows, monkeypatch):
    t, quantized, weight = inputs(points=points, fluxes=1, seed=13)
    p = torch.as_tensor(np.linspace(200.0, 399.0, rows), dtype=torch.float64, device="cuda")
    for bins in (4096, 512):  # the bin count changes between two calls on the same arrays
        sums, counts = core.accumulate_many(t, p, bins, quantized, weight, shared_weight=True)
        reference_sums, reference_counts = triton_result(monkeypatch, t, p, bins, quantized, weight, shared_weight=True)
        assert torch.equal(sums, reference_sums) and torch.equal(counts, reference_counts), f"{bins} bins"


@needs_rocm
def test_a_single_flux_whose_weights_are_too_large_for_the_counts_of_a_block_still_goes_to_the_triton_kernel():
    t, quantized, _ = inputs(points=2000, fluxes=1, seed=5)
    p = torch.as_tensor(np.linspace(5.0, 60.0, 9), dtype=torch.float64, device="cuda")
    weight = torch.full_like(quantized, 2**20)
    assert hip.accumulate_many(t, p, 512, quantized, weight, shared_weight=True) is None


def test_the_hip_accumulation_is_called_inside_the_device_context_of_the_time_axis(monkeypatch):
    """core.accumulate_many called the HIP kernel outside `on_device(t.device)`, and the kernel launches on the stream of the current device while its tensors are on t.device: with device='cuda:1' on a host of two GPUs it would run on device 0 over the pointers of device 1. The Triton kernel of the same function was already inside the context. Structural test, as one GPU is all that was available: the call is made while the context that `on_device` returned is open. Two devices were not tested."""
    inside = []

    class Context:
        def __enter__(self):
            inside.append(True)

        def __exit__(self, *exc):
            inside.pop()

    seen = []
    monkeypatch.setattr(core, "on_device", lambda device: Context())
    monkeypatch.setattr(hip, "accumulate_many", lambda *a, **k: seen.append(bool(inside)) or ("sums", "counts"))
    t = torch.zeros(4, dtype=torch.float64)
    found = core.accumulate_many(t, torch.ones(2, dtype=torch.float64), 8, torch.zeros((1, 4), dtype=torch.int64), torch.ones((1, 4), dtype=torch.int64), shared_weight=True)
    assert found == ("sums", "counts") and seen == [True], "the HIP kernel must launch with the device of t as the current device"


def entry_arguments(points=512, rows=3, fluxes=2, bins=64):
    """The tensors of a valid call of the C++ entry point (module.accumulate)."""
    t, quantized, weight = inputs(points=points, fluxes=fluxes)
    inverse = torch.full((rows,), 1.0 / 7.0, dtype=torch.float64, device="cuda")
    sums = torch.zeros(fluxes, rows, bins, dtype=torch.int64, device="cuda")
    counts = torch.zeros(rows, bins, dtype=torch.int64, device="cuda")
    return [t, inverse, quantized, weight[0].contiguous(), sums, counts, bins, 4096]


@needs_rocm
def test_the_entry_point_refuses_a_bin_count_outside_one_to_the_maximum_and_a_launch_that_cannot_run():
    """The entry point checked nothing: bins above MAXIMUM_BINS asked for more local memory than a block has, the launch failed, and the sums stayed zero with no error."""
    module = hip._load()
    assert module is not None
    module.accumulate(*entry_arguments())  # a valid call
    for bins in (0, -1, hip.MAXIMUM_BINS + 1, 100_000):
        arguments = entry_arguments(bins=max(bins, 1))
        arguments[6] = bins
        with pytest.raises(RuntimeError, match="bins"):
            module.accumulate(*arguments)


@needs_rocm
def test_the_entry_point_refuses_tensors_that_do_not_fit_the_points_and_a_time_axis_that_is_not_contiguous():
    module = hip._load()
    arguments = entry_arguments()
    with pytest.raises(RuntimeError, match="quantized"):
        module.accumulate(arguments[0], arguments[1], arguments[2][:, :-1].contiguous(), *arguments[3:])  # one point short: the kernel would read past it
    with pytest.raises(RuntimeError, match="weight"):
        module.accumulate(arguments[0], arguments[1], arguments[2], arguments[3][:-1].contiguous(), *arguments[4:])
    strided = torch.zeros(1024, dtype=torch.float64, device="cuda")[::2]
    with pytest.raises(RuntimeError, match="contiguous"):
        module.accumulate(strided, *arguments[1:])
    with pytest.raises(RuntimeError, match="device"):
        module.accumulate(arguments[0].cpu(), *arguments[1:])


@needs_rocm
def test_the_python_wrapper_makes_a_strided_time_axis_contiguous_and_the_bin_limit_is_the_one_of_the_source():
    t, quantized, weight = inputs(points=2000, fluxes=3)
    p = torch.as_tensor(np.linspace(3.0, 90.0, 12), dtype=torch.float64, device="cuda")
    wide = torch.zeros(4000, dtype=torch.float64, device="cuda")
    wide[::2] = t
    sums, counts = hip.accumulate_many(wide[::2], p, 256, quantized, weight, shared_weight=True)
    reference_sums, reference_counts = hip.accumulate_many(t, p, 256, quantized, weight, shared_weight=True)
    assert torch.equal(sums, reference_sums) and torch.equal(counts, reference_counts)


def test_the_extension_is_built_from_the_hashed_copy_of_the_source_text_and_not_from_the_path_of_the_file(monkeypatch, tmp_path):
    """The test of hip._source_path alone would stay green if _load went back to `sources=[str(SOURCE)]`: it checks the file that _load hands to torch's build."""
    from torch.utils import cpp_extension

    calls = []
    monkeypatch.setattr(hip, "_module", None)
    monkeypatch.setattr(hip, "_failed", False)
    monkeypatch.setattr(hip, "_built_for", None)
    monkeypatch.setattr(hip, "available", lambda: True)
    monkeypatch.setattr(hip, "_build_directory", lambda: tmp_path)
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100")
    monkeypatch.setattr(cpp_extension, "load", lambda **arguments: calls.append(arguments) or object())
    assert hip._load() is not None and len(calls) == 1
    (source,) = calls[0]["sources"]
    assert Path(source).parent == tmp_path and Path(source).name.startswith("accumulate-") and Path(source).read_text() == hip.SOURCE_TEXT, source
    assert Path(source) != hip.SOURCE


def test_the_bin_limit_of_the_source_is_the_one_of_hip_py():
    """Not marked for ROCm: the CPU run checks that the C++ entry point and hip.accumulate_many refuse the same number of bins."""
    assert f"#define MAXIMUM_BINS {hip.MAXIMUM_BINS}\n" in hip.SOURCE.read_text(), "the limit of the C++ entry point is the one of hip.py"
