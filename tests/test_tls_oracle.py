"""transitgpu.tls "gpu" against the brute-force oracle of TLS 2.0.1's test of its core (golden_upstream/core_oracle.py: for every duration and every phase shift the model of the folded light curve is built and its chi square summed).

Tolerances: the chi square of a period a relative 1e-9, the depth a relative 1e-9, and the template row equal. These are the tolerances that TLS's test applies to its own exact kernels (test_fused_kernel_matches_brute_force). The cases are those of that test: seeds 0 to 2, phase-shift margins 0, 0.01 and 0.05, equal and unequal errors, the default template and the eccentric square-root template (shorter than its window by one sample), and the periods 2.3, 1.7, 4.11, 0.93 and 0.61 days.
Function tests: the reference is TLS's oracle, run here."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent / "golden_upstream"))
oracle = pytest.importorskip("core_oracle")
import transitleastsquares.tls_constants as C  # noqa: E402
from transitleastsquares.backends import SearchProblem  # noqa: E402

from transitgpu.tls.window import WindowBackend  # noqa: E402

pytestmark = pytest.mark.gpu
PERIODS = [2.3, 1.7, 4.11, 0.93, 0.61]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("margin", [0.0, 0.01, 0.05])
@pytest.mark.parametrize("with_dy", [True, False])
@pytest.mark.parametrize("template", list(oracle.TEMPLATES))
def test_the_exact_backend_gives_the_chi_square_row_and_depth_of_the_brute_force_oracle(seed, margin, with_dy, template):
    if not torch.cuda.is_available():
        pytest.skip("no device: transitgpu.tls searches on a GPU only")
    t, y, dy = oracle.make_case(seed, n=900, with_dy=with_dy)
    overview, lc_arr = oracle.cache(t, y, template)
    limits = (C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX)
    problem = SearchProblem(t, y, dy, lc_arr, overview, 1e-5, *limits, margin)
    found = WindowBackend().search(problem, np.array(PERIODS))
    order = {float(period): index for index, period in enumerate(found.periods)}
    for period in PERIODS:
        expected_chi2, expected_row, expected_depth = oracle.brute_force_period(period, t, y, dy, 1e-5, lc_arr, overview, *limits, margin)
        index = order[period]
        np.testing.assert_allclose(found.chi2[index], expected_chi2, rtol=1e-9, err_msg=f"chi square of {period} d")
        assert found.rows[index] == expected_row, f"template row of {period} d"
        np.testing.assert_allclose(found.depths[index], expected_depth, rtol=1e-9, err_msg=f"depth of {period} d")
