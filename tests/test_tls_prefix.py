"""The folded flux and the prefix sums of the screened search (transitgpu.tls.prefix): the torch version against a plain numpy definition, and on a device the kernel against the torch version. Function tests: the inputs are made here."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu.tls.prefix import CHUNK, prefix_arrays, prefix_arrays_torch  # noqa: E402


def case(points, extra, rows=3, seed=1, device="cpu"):
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(1 + 0.001 * rng.normal(size=points), device=device)
    order = torch.stack([torch.as_tensor(rng.permutation(points), device=device) for _ in range(rows)])
    return y, order, 1.0 / 0.001**2, 123.4, extra


def numpy_definition(y, order, weight, mean, extra):
    y, order = y.cpu().numpy(), order.cpu().numpy()
    out = []
    for row in order:
        flux = y[row]
        patched = np.concatenate([flux, flux[:extra]])
        single = np.concatenate([[0.0], np.cumsum(weight * (1 - patched) - mean)])
        out.append((patched, np.concatenate([[0.0], np.cumsum(patched)]), np.concatenate([[0.0], np.cumsum(weight * (patched - 1) ** 2)]), np.concatenate([[0.0], np.cumsum(single)]), weight * np.sum((1 - flux) ** 2)))
    return [np.stack([o[i] for o in out]) for i in range(5)]


@pytest.mark.parametrize("points,extra", [(50, 7), (1, 0), (CHUNK, 5), (3 * CHUNK + 17, 400)])
def test_the_torch_version_is_the_numpy_definition_of_the_four_sums_and_the_constant(points, extra):
    arrays = prefix_arrays_torch(*case(points, extra))
    for got, expected in zip(arrays, numpy_definition(*case(points, extra))):
        assert np.allclose(got.numpy(), expected, rtol=1e-12, atol=1e-9)


@pytest.mark.gpu
@pytest.mark.parametrize("points,extra", [(50, 7), (1, 0), (CHUNK, 5), (CHUNK - 1, 1), (3 * CHUNK + 17, 400), (20000, 3000)])
def test_the_kernel_gives_the_sums_of_the_torch_version_to_rounding_including_chunk_edges_and_the_wrap(points, extra):
    if not torch.cuda.is_available():
        pytest.skip("no device")
    arguments = case(points, extra, device="cuda")
    for got, expected in zip(prefix_arrays(*arguments), prefix_arrays_torch(*arguments)):
        assert got.shape == expected.shape
        scale = float(expected.abs().max()) + 1.0
        assert float((got - expected).abs().max()) < 1e-12 * scale * max(points, 1), "sums in another order differ by rounding only"


@pytest.mark.gpu
def test_the_sums_of_a_period_do_not_depend_on_the_other_periods_of_the_batch_over_many_chunks():
    """torch.cumsum along the last dimension can sum in a different order for 64 rows than for 128; the carries are
    summed in sequence, so the sums of a period do not depend on the batch."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    y, order, weight, mean, extra = case(100_000, 3000, rows=128, device="cuda")
    small = prefix_arrays(y, order[:64], weight, mean, extra)
    large = prefix_arrays(y, order, weight, mean, extra)
    for name, a, b in zip(("patched", "cumulative", "squares", "double", "constant"), small, large):
        assert torch.equal(a, b[:64]), f"{name} of the first 64 periods differs between a batch of 64 and one of 128"


@pytest.mark.gpu
@pytest.mark.parametrize("points,extra", [(50, 7), (1, 0), (CHUNK, 5), (3 * CHUNK + 17, 400)])
def test_the_residual_variant_returns_the_weighted_residual_of_the_flux_in_place_of_the_flux_and_the_same_sums(points, extra):
    """With residual=True the first array is weight * (1 - patched), rounded as the dot kernel rounds it, and the other four are those of the default call."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    y, order, weight, mean, extra = case(points, extra, device="cuda")
    patched, *sums = prefix_arrays(y, order, weight, mean, extra)
    residual, *same = prefix_arrays(y, order, weight, mean, extra, residual=True)
    assert residual.shape == patched.shape and torch.equal(residual, weight * (1.0 - patched))
    for name, got, expected in zip(("cumulative", "squares", "double", "constant"), same, sums):
        assert torch.equal(got, expected), name
