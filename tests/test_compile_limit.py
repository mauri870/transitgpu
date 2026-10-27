"""A process that searches with every combination of options that production uses gives the same numbers on every pass.

torch keeps 8 compiled variants of a function and then runs it eagerly, which changes the float32 statistic in its seventh digit; the package raises the limit (core.RECOMPILE_LIMIT) so that no variant of a long process is dropped. A function test of two passes over the same searches, one with room for every compiled variant and one at torch's default limit, which must agree to the bit.
"""

import itertools

import numpy as np
import pytest

pytest.importorskip("torch")

from transitgpu import core
from transitgpu.core import search_many

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="the scorer is compiled on a device only; a CPU runs it eagerly in every process")]


def light_curve():
    time = np.arange(0.0, 60.0, 1 / 24)
    flux = 1 + np.random.default_rng(8).normal(0, 500e-6, time.size)
    flux[np.abs(time - (2.0 + np.round((time - 2.0) / 7.3) * 7.3)) < 2 / 24] -= 2000e-6
    return time, flux


def test_every_combination_of_options_of_a_production_process_is_searched_to_the_same_bits_in_any_order(monkeypatch):
    import torch

    name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    time, flux = light_curve()
    weights = np.random.default_rng(9).uniform(0.5, 2.0, time.size)
    shuffles = [np.roll(flux, shift) for shift in (50, 200, 400)]
    combinations = list(itertools.product((False, True), ("float32", "float64"), (None, weights), (False, True), (False, True)))

    def run(combination):
        bumps, precision, given, many, spectrum = combination
        return search_many(time, [flux] + (shuffles if many else []), 0.6, 10.0, spectrum=spectrum, bumps=bumps, precision={"statistic": precision}, weights=given, cuts=[5.0, 10.0])

    def same(one, other):
        assert type(one) is type(other)
        if isinstance(one, list):
            assert len(one) == len(other)
            for first, second in zip(one, other):
                same(first, second)
        elif isinstance(one, dict):
            assert one.keys() == other.keys()
            for key in one:
                same(one[key], other[key])
        else:
            np.testing.assert_array_equal(one, other)

    # the reference: every variant compiled, with room for all of them
    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, name, 512)
    reference = [run(combination) for combination in combinations]
    # a production process: torch's default limit, which the package raises, over the same searches
    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, name, 8)
    production = [run(combination) for combination in combinations]
    for first, second in zip(reference, production):
        same(first, second)
    assert getattr(torch._dynamo.config, name) >= core.RECOMPILE_LIMIT, "the package raised torch's limit"


def test_a_limit_that_the_caller_set_higher_is_not_lowered(monkeypatch):
    import torch

    name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    monkeypatch.setattr(torch._dynamo.config, name, core.RECOMPILE_LIMIT * 4)
    core.raise_recompile_limit()
    assert getattr(torch._dynamo.config, name) == core.RECOMPILE_LIMIT * 4
    monkeypatch.setattr(torch._dynamo.config, name, 2)
    core.raise_recompile_limit()
    assert getattr(torch._dynamo.config, name) == core.RECOMPILE_LIMIT
