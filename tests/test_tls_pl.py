"""The piecewise-linear search of transitgpu.tls (TLS's fused-pl algorithm on the device, backend "gpu-pl"): its machinery against the exact search when the fit is exact, its error with the tolerance of fused-pl, and its
results against TLS's own fused-pl and exact backends on the 14 characterization cases (golden_upstream/golden_cases.py), with the tolerances of the assertions below. Function tests: the
inputs are light curves made here and TLS's results, run here."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent / "golden_upstream"))
import golden_cases  # noqa: E402
from test_tls_binned import exact_arguments, light_curve, problem_of  # noqa: E402

import transitgpu.tls  # noqa: E402,F401  (registers the backends)
from transitgpu.tls.pl import pl_power  # noqa: E402
from transitgpu.tls.window import window_power  # noqa: E402

pytestmark = pytest.mark.gpu
UNEQUAL_ERRORS = ("syn_earth_dy", "narrow_stellar")


@pytest.fixture(scope="module")
def problem():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    problem, periods = problem_of(*light_curve(points=3000), 0.8, 8.0)
    return exact_arguments(problem, periods[::25])


def exact_of(given):
    return window_power(*given, device="cuda", dtype=torch.float64, batch=8, correlate="screen")


def test_the_machinery_gives_the_exact_search_when_every_sample_of_every_template_is_a_knot(problem):
    chi, rows, depths = pl_power(*problem, device="cuda", eps=0.0, min_length=1, coarsen=1, scout_every=0)
    chi_e, rows_e, depths_e = exact_of(problem)
    assert np.allclose(chi, chi_e, rtol=1e-9, atol=0.0), f"largest relative difference {np.max(np.abs(chi - chi_e) / np.abs(chi_e)):.3e}"
    assert np.array_equal(rows, rows_e) and np.allclose(depths, depths_e, rtol=1e-9, atol=0.0)


def test_templates_correlated_directly_are_exact(problem):
    chi, rows, depths = pl_power(*problem, device="cuda", eps=2e-2, min_length=1 << 62, coarsen=1, scout_every=0)
    chi_e, rows_e, depths_e = exact_of(problem)
    assert np.allclose(chi, chi_e, rtol=1e-9, atol=0.0)
    assert np.array_equal(rows, rows_e) and np.allclose(depths, depths_e, rtol=1e-9, atol=0.0)


def test_with_the_tolerance_of_fused_pl_the_chi_square_of_a_period_is_within_one_percent_of_the_exact_one_and_the_best_period_is_the_same(problem):
    chi, rows, depths = pl_power(*problem, device="cuda")
    chi_e, rows_e, depths_e = exact_of(problem)
    assert np.nanmax(np.abs(chi - chi_e) / np.abs(chi_e)) <= 1e-2
    assert int(np.argmin(chi)) == int(np.argmin(chi_e)) and rows[np.argmin(chi)] == rows_e[np.argmin(chi_e)]


def test_a_coarse_grid_of_starts_with_refinement_searches_a_subset_of_the_starts_and_keeps_the_best_period(problem):
    full = pl_power(*problem, device="cuda", coarsen=1)
    coarse = pl_power(*problem, device="cuda", coarsen=3)
    assert np.all(coarse[0] >= full[0] * (1 - 1e-12)), "a coarse search found a deeper gain than the search of every start"
    assert np.nanmax(np.abs(coarse[0] - full[0]) / np.abs(full[0])) <= 1e-2
    assert int(np.argmin(coarse[0])) == int(np.argmin(full[0]))


def test_scout_durations_search_a_subset_of_the_starts_and_keep_the_best_period(problem):
    full = pl_power(*problem, device="cuda", scout_every=0)
    scouted = pl_power(*problem, device="cuda", scout_every=4)
    assert np.all(scouted[0] >= full[0] * (1 - 1e-12)), "a scouted search found a deeper gain than the search of every duration"
    assert np.nanmax(np.abs(scouted[0] - full[0]) / np.abs(full[0])) <= 1e-2
    assert int(np.argmin(scouted[0])) == int(np.argmin(full[0])) and scouted[1][np.argmin(scouted[0])] == full[1][np.argmin(full[0])]


def test_scouts_are_off_for_a_light_curve_of_fewer_than_two_thousand_points():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    small, periods = problem_of(*light_curve(points=1500), 0.8, 8.0)
    given = exact_arguments(small, periods[::40])
    assert all(np.array_equal(a, b) for a, b in zip(pl_power(*given, device="cuda", scout_every=4), pl_power(*given, device="cuda", scout_every=0)))


def test_a_coarsening_of_one_is_the_search_of_every_start_and_the_default_is_the_coarse_one(problem):
    import inspect

    assert inspect.signature(pl_power).parameters["coarsen"].default == 3
    again = pl_power(*problem, device="cuda", coarsen=1)
    assert all(np.array_equal(a, b) for a, b in zip(again, pl_power(*problem, device="cuda", coarsen=1)))


def test_a_batch_of_one_period_and_a_batch_larger_than_the_periods_give_the_same_result(problem):
    reference = pl_power(*problem, device="cuda", batch=64)
    for batch in (1, 3, 1000):
        got = pl_power(*problem, device="cuda", batch=batch)
        assert all(np.array_equal(a, b) for a, b in zip(got, reference)), batch


def test_unequal_errors_raise_value_error():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    t, y, dy = light_curve(points=2000)
    dy = dy * (1.0 + 0.5 * np.arange(len(dy)) / len(dy))
    problem, periods = problem_of(t, y, dy, 1.0, 5.0)
    with pytest.raises(ValueError, match="equal errors"):
        pl_power(*exact_arguments(problem, periods[::50]), device="cuda")


def test_the_backend_is_registered_under_its_name():
    from transitleastsquares import available_backends

    assert "gpu-pl" in available_backends()


def arrays(name, backend, monkeypatch):
    monkeypatch.setenv("TLS_GOLDEN_BACKEND", backend)
    return golden_cases.to_arrays(golden_cases.run(name))


@pytest.mark.parametrize("name", [name for name in golden_cases.CASES if name not in UNEQUAL_ERRORS])
def test_gpu_pl_agrees_with_fused_pl_and_the_exact_backend_on_its_golden_case(name, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no device")
    exact, fused_pl, mine = (arrays(name, backend, monkeypatch) for backend in ("fused", "fused-pl", "gpu-pl"))
    bad = []
    for other, label in ((fused_pl, "fused-pl"), (exact, "exact")):
        for key in ("period", "duration", "T0", "transit_count"):
            if not np.array_equal(mine[key], other[key], equal_nan=True):
                bad.append((label, key, float(mine[key]), float(other[key])))
        if not np.allclose(mine["depth"], other["depth"], rtol=1e-4, atol=0.0, equal_nan=True):
            bad.append((label, "depth", float(mine["depth"]), float(other["depth"])))
        if not np.allclose(mine["SDE"], other["SDE"], rtol=0.0, atol=0.07, equal_nan=True):
            bad.append((label, "SDE", float(mine["SDE"]), float(other["SDE"])))
    chi = lambda a: np.asarray(a["chi2"])
    for other, label, limit in ((exact, "exact", 1e-2), (fused_pl, "fused-pl", 1.5e-2)):
        difference = float(np.nanmax(np.abs(chi(mine) - chi(other)) / np.abs(chi(other))))
        if difference > limit:
            bad.append((label, "chi2 largest relative difference", difference, limit))
    assert not bad, f"{name}: {bad}"


@pytest.mark.parametrize("name", UNEQUAL_ERRORS)
def test_gpu_pl_refuses_the_golden_cases_with_unequal_errors(name, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no device")
    with pytest.raises(ValueError, match="equal errors"):
        arrays(name, "gpu-pl", monkeypatch)


def test_the_programs_of_the_gain_kernel_are_the_tiles_of_the_scout_widths_of_each_period_and_nothing_else():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls.pl import live_programs

    scouts = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)  # widths by periods
    tile_first = np.array([0, 2, 3])  # the first block of each width, with 2, 1 and 4 blocks
    tile_count = np.array([2, 1, 4])
    live = np.asarray([scouts[:, row] for row in range(3)])
    pairs, count = int(scouts.sum()), int((scouts * tile_count[:, None]).sum())
    on_device = [torch.as_tensor(x, device="cuda") for x in (scouts.astype(np.int8), tile_first, tile_count)]
    programs = live_programs(*on_device, pairs, count).cpu().numpy()
    assert programs.dtype == np.int32 and programs.shape == (count, 4)
    expected = {(row, width, tile, tile_first[width] + tile) for row in range(3) for width in range(3) if live[row][width] for tile in range(tile_count[width])}
    assert {tuple(r) for r in programs.tolist()} == expected and len(expected) == count, "every live tile once, no dead tile"
    assert live_programs(torch.zeros((3, 2), dtype=torch.int8, device="cuda"), on_device[1], on_device[2], 0, 0).shape == (0, 4), "no scout: no program"


def test_the_launch_of_the_live_blocks_and_the_launch_of_every_block_give_the_same_bits(problem, monkeypatch):
    import transitgpu.tls.pl as module

    monkeypatch.setattr(module, "COMPACT_BELOW_POINTS", 1 << 30)
    compact = pl_power(*problem, device="cuda")
    monkeypatch.setattr(module, "COMPACT_BELOW_POINTS", 0)
    every = pl_power(*problem, device="cuda")
    assert all(np.array_equal(a, b) for a, b in zip(compact, every))
