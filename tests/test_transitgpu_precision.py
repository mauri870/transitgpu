"""The precision of the box search: the statistic in float32 against float64, and the integer sums of the one-pass
accumulation kernels against the fold table and scatter_add_, bit for bit. Function tests: two paths agree on a flux."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu import core as gpu  # noqa: E402
from transitgpu.core import search_many  # noqa: E402

from test_transitgpu import light_curve  # noqa: E402

pytestmark = pytest.mark.gpu  # every test of this module may search on the device


def spectra(precision, fluxes=None, **options):
    time, flux = light_curve(seed=3, cadence_min=30.0)
    return search_many(time, fluxes if fluxes is not None else [flux], 0.5, 14, spectrum=True, precision=precision, **options)


def table_spectra(fluxes=None, time=None):
    """The spectra of the default precision with the one-pass accumulation turned off: the fold table and scatter_add_."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gpu, "_accumulate_many_kernel", lambda: False)
        if time is None:
            return spectra({}, fluxes=fluxes)
        return search_many(time, fluxes, 0.5, 14, spectrum=True)


def test_the_statistic_in_32_bits_gives_the_same_best_period_and_a_spectrum_within_a_small_relative_error():
    exact, single = spectra({"statistic": "float64"})[0], spectra({"statistic": "float32"})[0]
    assert exact["period_days"] == single["period_days"]
    assert np.max(np.abs(single["power"] - exact["power"]) / np.maximum(np.abs(exact["power"]), 1.0)) < 1e-5
    assert abs(single["sde"] / exact["sde"] - 1) < 1e-5


def test_the_default_precision_is_the_statistic_in_32_bits_and_the_rest_as_the_search_was_and_float64_is_the_option_that_gives_the_earlier_numbers():
    """The statistic in float32 is the default; `precision={"statistic": "float64"}` computes it in float64."""
    time, flux = light_curve(seed=3, cadence_min=30.0)
    default = search_many(time, [flux], 0.5, 14, spectrum=True)[0]
    assert gpu.PRECISION == {"statistic": "float32"}
    assert np.array_equal(default["power"], spectra({"statistic": "float32"})[0]["power"])
    assert np.array_equal(default["power"], table_spectra()[0]["power"]), "the fold table and the scatters give the same spectrum to the bit"
    earlier = spectra({"statistic": "float64"})[0]
    assert default["period_days"] == earlier["period_days"] and not np.array_equal(default["power"], earlier["power"])


def test_an_unknown_precision_is_refused_by_name():
    with pytest.raises(ValueError, match="precision"):
        spectra({"statistic": "float16"})
    with pytest.raises(ValueError, match="precision.*'statistic'"):
        spectra({"sums": "int64"})
    with pytest.raises(ValueError, match="precision"):
        spectra({"nonsense": "yes"})


def torch_sums(time_axis, periods, bins, quantized, weight):
    """The sums and counts of a batch by the fold table and two scatter-adds."""
    t = torch.as_tensor(time_axis, dtype=torch.float64, device="cuda")
    p = torch.as_tensor(periods, dtype=torch.float64, device="cuda")
    table = gpu.fold_index(t, p, bins)
    q = torch.as_tensor(quantized, dtype=torch.int64, device="cuda")
    w = torch.as_tensor(weight, dtype=torch.int64, device="cuda")
    rows = len(periods)
    return (torch.zeros(rows, bins, dtype=torch.int64, device="cuda").scatter_add_(1, table, q.unsqueeze(0).expand(rows, q.numel())),
            torch.zeros(rows, bins, dtype=torch.int64, device="cuda").scatter_add_(1, table, w.unsqueeze(0).expand(rows, w.numel())))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("bins", [128, 1024, 4096])
def test_the_fused_fold_and_accumulation_gives_the_sums_and_counts_of_the_fold_table_and_the_scatters_to_the_bit(bins):
    """The bin of each point is computed in the thread that adds the point, with the operations of the torch fold, and integer additions do not depend on their order."""
    generator = np.random.default_rng(bins)
    time_axis = np.sort(generator.uniform(0.0, 2800.0, 6000))
    periods = np.sort(np.concatenate([generator.uniform(0.5, 14.0, 9), [1.0, 2.0 * 2 / 1440, 3.18]]))  # a period that is a multiple of the cadence puts many points in few bins
    quantized = np.round(generator.normal(0.0, 3e5, time_axis.size)).astype(np.int64)
    quantized[::997] = 9 * 10**8  # outliers, as large as the 32-bit bound allows a few of
    weight = (generator.uniform(size=time_axis.size) > 0.1).astype(np.int64)
    expected = torch_sums(time_axis, periods, bins, quantized, weight)
    found = gpu.accumulate(torch.as_tensor(time_axis, dtype=torch.float64, device="cuda"), torch.as_tensor(periods, dtype=torch.float64, device="cuda"), bins,
                           torch.as_tensor(quantized, dtype=torch.int64, device="cuda"), torch.as_tensor(weight, dtype=torch.int64, device="cuda"))
    assert torch.equal(found[0], expected[0]) and torch.equal(found[1], expected[1])


def test_a_time_axis_with_negative_times_keeps_the_torch_path_and_gives_the_same_numbers(monkeypatch):
    """The one-pass accumulation needs non-negative times; with negative times the search takes the fold table."""
    time, flux = light_curve(seed=3, cadence_min=30.0)
    shifted = time - 5.0
    calls = spy_on_accumulate(monkeypatch)
    plain = search_many(shifted, [flux], 0.5, 14, spectrum=True)[0]
    assert not calls, "no one-pass accumulation for negative times"
    assert np.array_equal(plain["power"], table_spectra([flux], time=shifted)[0]["power"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("bins, fluxes", [(128, 1), (1024, 5), (4096, 12)])
def test_one_pass_over_the_fluxes_of_a_batch_gives_the_sums_and_counts_of_the_table_and_the_scatters_to_the_bit(bins, fluxes):
    """Each thread computes a point's bin once and adds that point of every flux of the batch to it."""
    generator = np.random.default_rng(bins + fluxes)
    time_axis = np.sort(generator.uniform(0.0, 2800.0, 6000))
    periods = np.sort(np.concatenate([generator.uniform(0.5, 14.0, 9), [1.0, 2.0 * 2 / 1440, 3.18]]))
    quantized = np.round(generator.normal(0.0, 3e5, (fluxes, time_axis.size))).astype(np.int64)
    quantized[:, ::997] = 9 * 10**8
    weight = (generator.uniform(size=(fluxes, time_axis.size)) > 0.1).astype(np.int64)  # the points without data differ from flux to flux
    t = torch.as_tensor(time_axis, dtype=torch.float64, device="cuda")
    p = torch.as_tensor(periods, dtype=torch.float64, device="cuda")
    sums, counts = gpu.accumulate_many(t, p, bins, torch.as_tensor(quantized, device="cuda"), torch.as_tensor(weight, device="cuda"))
    assert sums.shape == (fluxes, len(periods), bins)
    for k in range(fluxes):
        expected = torch_sums(time_axis, periods, bins, quantized[k], weight[k])
        assert torch.equal(sums[k], expected[0]) and torch.equal(counts[k], expected[1]), k


def test_the_search_of_many_fluxes_in_one_pass_gives_the_spectra_of_the_torch_path_to_the_bit():
    time, flux = light_curve(seed=3, cadence_min=30.0)
    shuffled = np.concatenate([flux[100:], flux[:100]])
    gapped = flux.copy()
    gapped[200:260] = np.nan
    fluxes = [flux, shuffled, gapped]
    plain, batch = table_spectra(fluxes), spectra({}, fluxes=fluxes)
    assert all(np.array_equal(a["power"], b["power"]) and a["period_days"] == b["period_days"] for a, b in zip(plain, batch))


def test_the_one_pass_in_chunks_of_fewer_fluxes_than_the_batch_gives_the_same_spectra(monkeypatch):
    """Three fluxes in chunks of two (a chunk and a remainder) give the spectra of one chunk of three."""
    time, flux = light_curve(seed=3, cadence_min=30.0)
    fluxes = [flux, np.concatenate([flux[100:], flux[:100]]), np.concatenate([flux[300:], flux[:300]])]
    plain = spectra({}, fluxes=fluxes)
    monkeypatch.setattr(gpu, "FLUX_CHUNK", 2)
    chunked = spectra({}, fluxes=fluxes)
    assert all(np.array_equal(a["power"], b["power"]) for a, b in zip(plain, chunked))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("bins, fluxes", [(128, 1), (1024, 5), (4096, 12)])
def test_the_counts_added_once_for_fluxes_with_equal_weights_give_the_sums_and_counts_of_the_torch_path_to_the_bit(bins, fluxes):
    """When every flux of a batch has the same weights, the counts are added for the first flux only and repeated; the integers are those of the table and the scatters."""
    generator = np.random.default_rng(bins + fluxes + 1)
    time_axis = np.sort(generator.uniform(0.0, 2800.0, 6000))
    periods = np.sort(np.concatenate([generator.uniform(0.5, 14.0, 9), [1.0, 3.18]]))
    quantized = np.round(generator.normal(0.0, 3e5, (fluxes, time_axis.size))).astype(np.int64)
    weight_row = (generator.uniform(size=time_axis.size) > 0.1).astype(np.int64)
    weight = np.broadcast_to(weight_row, (fluxes, time_axis.size)).copy()
    t = torch.as_tensor(time_axis, dtype=torch.float64, device="cuda")
    p = torch.as_tensor(periods, dtype=torch.float64, device="cuda")
    sums, counts = gpu.accumulate_many(t, p, bins, torch.as_tensor(quantized, device="cuda"), torch.as_tensor(weight, device="cuda"), shared_weight=True)
    assert sums.shape == counts.shape == (fluxes, len(periods), bins)
    for k in range(fluxes):
        expected = torch_sums(time_axis, periods, bins, quantized[k], weight[k])
        assert torch.equal(sums[k], expected[0]) and torch.equal(counts[k], expected[1]), k


def spy_on_accumulate(monkeypatch):
    calls = []
    original = gpu.accumulate_many
    monkeypatch.setattr(gpu, "accumulate_many", lambda *arguments, **options: calls.append(options.get("shared_weight", False)) or original(*arguments, **options))
    return calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_batch_whose_fluxes_have_the_same_points_adds_the_counts_once_and_a_batch_with_a_different_gap_does_not(monkeypatch):
    """Fluxes on the same points share their weights (the shuffles of a star) and a flux with a gap of its own does not;
    the check is made once per search. Either way the spectra are the torch path's to the bit."""
    time, flux = light_curve(seed=3, cadence_min=30.0)
    shuffled = np.concatenate([flux[100:], flux[:100]])
    gapped = flux.copy()
    gapped[200:260] = np.nan
    calls = spy_on_accumulate(monkeypatch)
    same = [flux, shuffled]
    plain, batch = table_spectra(same), spectra({}, fluxes=same)
    assert calls and all(calls), "equal weights: every launch passes shared_weight"
    assert all(np.array_equal(a["power"], b["power"]) for a, b in zip(plain, batch))
    calls.clear()
    different = [flux, gapped]
    plain, batch = table_spectra(different), spectra({}, fluxes=different)
    assert calls and not any(calls), "a flux with its own gap: no launch passes shared_weight"
    assert all(np.array_equal(a["power"], b["power"]) for a, b in zip(plain, batch))


def test_the_kernels_run_with_their_own_device_current_and_the_name_is_that_of_the_device_in_use(monkeypatch):
    """Triton launches on the current device, so every launch must enter the device of its tensors, and device_name()
    must name the device given. Checked with stand-ins for torch.cuda.device and for a second device's name."""
    from transitgpu import core

    if not torch.cuda.is_available():
        pytest.skip("no device: the test enters the CUDA device of a tensor")
    entered = []

    class Recorder:
        def __init__(self, device):
            entered.append(torch.device(device))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(torch.cuda, "device", Recorder)
    t = torch.linspace(0.0, 100.0, 5000, dtype=torch.float64, device="cuda")
    periods = torch.tensor([3.18, 3.19], dtype=torch.float64, device="cuda")
    quantized = torch.ones(5000, dtype=torch.int64, device="cuda")
    core.accumulate(t, periods, 32, quantized, torch.ones_like(quantized))
    core.accumulate_many(t, periods, 32, quantized[None, :], torch.ones_like(quantized)[None, :])
    assert entered and all(device.type == "cuda" for device in entered) and len(entered) >= 2, entered
    monkeypatch.undo()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: f"gpu {index}")
    assert core.device_name() == "gpu 1" and core.device_name(torch.device("cuda:0")) == "gpu 0"
