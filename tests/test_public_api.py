"""The public searches of transitgpu: how they fail without a GPU and without CUDA, what the engine in transitgpu.core does where they refuse, and what the docstrings of the searches name. Function tests: they check names, messages and the callable surface, and none searches on a device (the device is hidden from torch where a test needs it absent).
"""
import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve  # noqa: E402

import transitgpu  # noqa: E402
from transitgpu import core  # noqa: E402

RESULT_FIELDS = ("depth_mean_ppm", "depth_ppm", "duration_corrected_hr", "duration_hr", "fill_factor", "period_days", "period_max_searched_days", "period_min_searched_days", "period_uncertainty_days", "periods", "power",
                 "sde", "snr", "t0_days", "transit_count")


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def calls(time, flux):
    return {"search": lambda **options: transitgpu.search(time, flux, 0.5, 14.0, **options), "search_many": lambda **options: transitgpu.search_many(time, [flux], 0.5, 14.0, **options)}


def test_the_searches_of_the_package_raise_without_a_device_or_for_the_cpu_and_name_astropys_box_least_squares(no_gpu):
    time, flux = light_curve(2000, 60.0)
    for name, call in calls(time, flux).items():
        for options in ({}, {"device": "cpu"}):
            with pytest.raises(RuntimeError, match="BoxLeastSquares"):
                call(**options)


def test_a_cuda_device_asked_for_where_torch_finds_none_raises_and_says_how_to_install_torch(no_gpu):
    time, flux = light_curve(2000, 60.0)
    with pytest.raises(RuntimeError, match=r"no CUDA or ROCm device.*docs\.astral\.sh/uv/guides/integration/pytorch"):
        transitgpu.box_search(time, flux).power(period_min=0.5, period_max=14, device="cuda")
    for name, call in calls(time, flux).items():
        with pytest.raises(RuntimeError, match="no CUDA or ROCm device"):
            call(device="cuda")


def test_the_engine_in_core_searches_on_a_cpu_where_the_public_searches_refuse(no_gpu):
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


def test_a_device_passed_by_position_is_checked_like_one_passed_by_name(monkeypatch):
    """torch is made to report a GPU, so only the device "cpu" given by position can make the call raise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    time, flux = light_curve(2000, 60.0)
    with pytest.raises(RuntimeError, match="BoxLeastSquares"):
        transitgpu.search_many(time, [flux], 0.5, 14.0, False, "cpu")
    with pytest.raises(RuntimeError, match="BoxLeastSquares"):
        transitgpu.search(time, flux, 0.5, 14.0, False, 1, "cpu")
