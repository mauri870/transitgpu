"""The public searches of transitgpu: that they run on the CPU without a GPU, how they fail for a CUDA device that torch does not find, and what the docstrings of the searches name. Function tests: they check names, messages and the callable surface, and none searches on a GPU (the device is hidden from torch where a test needs it absent).
"""
import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve

import transitgpu
from transitgpu import core

RESULT_FIELDS = ("depth_mean_ppm", "depth_ppm", "duration_corrected_hr", "duration_hr", "fill_factor", "period_days", "period_max_searched_days", "period_min_searched_days", "period_uncertainty_days", "periods", "power",
                 "sde", "snr", "t0_days", "transit_count")


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def calls(time, flux):
    return {"search": lambda **options: transitgpu.search(time, flux, 0.5, 14.0, **options), "search_many": lambda **options: transitgpu.search_many(time, [flux], 0.5, 14.0, **options)}


def test_without_a_gpu_the_searches_of_the_package_run_on_the_cpu_by_default_and_when_asked_and_give_the_engines_result(no_gpu):
    time, flux = light_curve()
    expected = core.search(time, flux, 0.5, 14.0, device="cpu")
    for name, call in calls(time, flux).items():
        for options in ({}, {"device": "cpu"}):
            found = call(**options)
            found = found[0] if name == "search_many" else found
            assert (found["period_days"], found["sde"]) == (expected["period_days"], expected["sde"]), (name, options)


def test_a_cuda_device_asked_for_where_torch_finds_none_raises_and_says_how_to_install_torch(no_gpu):
    time, flux = light_curve(2000, 60.0)
    with pytest.raises(RuntimeError, match=r"no CUDA or ROCm device.*docs\.astral\.sh/uv/guides/integration/pytorch"):
        transitgpu.box_search(time, flux).power(period_min=0.5, period_max=14, device="cuda")
    for name, call in calls(time, flux).items():
        with pytest.raises(RuntimeError, match="no CUDA or ROCm device"):
            call(device="cuda")


def test_the_engine_in_core_searches_on_a_cpu(no_gpu):
    time, flux = light_curve(2000, 60.0)
    assert core.search(time, flux, 0.5, 14.0)["period_days"] > 0
    assert len(core.search_many(time, [flux, flux], 0.5, 14.0)) == 2
    assert "not the public interface" in core.__doc__


def test_the_public_searches_are_the_engine_with_its_documentation_and_name():
    for name in ("search", "search_many"):
        public, engine = getattr(transitgpu, name), getattr(core, name)
        assert public.__wrapped__ is engine and public.__doc__ == engine.__doc__ and public.__name__ == name


def test_the_docstring_of_search_names_every_field_that_it_returns(no_gpu):
    """The fields are documented once, in search_many, and the docstring of search points there."""
    time, flux = light_curve(2000, 60.0)
    found = core.search(time, flux, 0.5, 14.0, spectrum=True)
    assert set(found) == set(RESULT_FIELDS), "the fields the test lists are the fields of the result"
    missing = [field for field in found if field not in core.search_many.__doc__]
    assert not missing, f"fields the docstring of search_many does not name: {missing}"
    assert "centers" in core.search_many.__doc__ and "widths_days" in core.search_many.__doc__
    assert "search_many" in core.search.__doc__


def test_a_device_passed_by_position_is_checked_like_one_passed_by_name(no_gpu):
    """torch is made to find no GPU, so the device "cuda" given by position must make the call raise before it searches."""
    time, flux = light_curve(2000, 60.0)
    with pytest.raises(RuntimeError, match="no CUDA or ROCm device"):
        transitgpu.search_many(time, [flux], 0.5, 14.0, False, "cuda")
    with pytest.raises(RuntimeError, match="no CUDA or ROCm device"):
        transitgpu.search(time, flux, 0.5, 14.0, False, 1, "cuda")
