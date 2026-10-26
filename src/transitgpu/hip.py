"""The accumulation of the box search in the local memory of the device, for ROCm.

`accumulate_many` has the arguments and the result of transitgpu.core.accumulate_many and returns None when it does not
apply: not on ROCm, no compiler (hipcc), weights that differ between fluxes or are too large for a block's counts, or
more than MAXIMUM_BINS bins. The caller then uses the Triton kernel. The source is compiled at the first use and cached
in a directory of the environment. The architecture of the build is that of the device, so that the build is the same in every process and is
found again, and a lock left by a build that was killed is waited on for LOCK_WAIT_SECONDS, after which the Triton kernel is used.
"""
import os
import re
import sys
import time
import warnings
from pathlib import Path

import torch

MAXIMUM_BINS = 4096  # the table of a block (8 bytes of sum and 4 of count a bin) must fit the 64 KiB of local memory
SOURCE = Path(__file__).with_name("accumulate.hip")
LOCK_WAIT_SECONDS = 120.0  # a build takes 20 to 32 s (measured, one or two architectures); a lock older than four builds is not a build that is running
_module = None
_failed = False
_built_for = None  # the architectures the extension was built for, or None when they are not known
_warned_architectures = set()


def available():
    return bool(torch.version.hip) and torch.cuda.is_available()


def _architecture():
    """The architecture of the current device without its features ("gfx1100" for "gfx1100:sramecc+:xnack-"), or None."""
    try:
        name = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except (AttributeError, RuntimeError, AssertionError):  # torch without a device raises an AssertionError
        return None
    return name.split(":")[0] or None


def _device_architecture(device):
    """The architecture of a device without its features, or None when it has none (a CPU) or it cannot be read."""
    try:
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] or None
    except (AttributeError, RuntimeError, ValueError, TypeError, AssertionError):
        return None


def _architectures(chosen, user):
    """The architectures of a build: the one chosen for it, else those of the list that the user set in PYTORCH_ROCM_ARCH (separated by semicolons, commas or spaces, features after a colon dropped), else None."""
    if chosen:
        return {chosen}
    if user:
        return {name.split(":")[0] for name in re.split(r"[;,\s]+", user) if name}
    return None


def _build_directory():
    """The directory of the build: one of the environment, so that two environments with different builds of torch do not rebuild in turn or wait on each other's lock; None, torch's own directory, when the user set
    TORCH_EXTENSIONS_DIR or the environment cannot be written to."""
    if os.environ.get("TORCH_EXTENSIONS_DIR"):
        return None
    directory = Path(sys.prefix) / "transitgpu_extension"
    return directory if os.access(directory.parent, os.W_OK) else None


def _wait_for_lock(directory, seconds):
    """True when the lock file of the build in `directory` is absent, waiting up to `seconds` for it to go; False when it is still there."""
    lock = Path(directory) / "lock" if directory is not None else _default_lock()
    deadline = time.monotonic() + seconds
    while lock is not None and lock.exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))
    return True


def _default_lock():
    from torch.utils import cpp_extension

    root = cpp_extension._get_build_directory("transitgpu_accumulate", verbose=False)
    return Path(root) / "lock"


def _load():
    global _module, _failed, _built_for
    if _module is None and not _failed:
        directory = _build_directory()
        lock = (directory / "lock") if directory is not None else None
        try:
            from torch.utils import cpp_extension

            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
            if not _wait_for_lock(directory, LOCK_WAIT_SECONDS):
                _failed = True
                where = lock if lock is not None else _default_lock()
                warnings.warn(f"the HIP accumulation is not available (the lock file {where} of a build was still there after {LOCK_WAIT_SECONDS:g} s, left by a build that was killed: remove it); the Triton kernel is used", stacklevel=3)
                return None
            user = os.environ.get("PYTORCH_ROCM_ARCH")
            chosen = _architecture() if user is None else None  # torch lists the architectures of the cards in an order that changes, which changed the build command and recompiled in every process
            if chosen:
                os.environ["PYTORCH_ROCM_ARCH"] = chosen
            try:
                _module = cpp_extension.load(name="transitgpu_accumulate", sources=[str(SOURCE)], build_directory=None if directory is None else str(directory))
                _built_for = _architectures(chosen, user)
            finally:
                if chosen:
                    del os.environ["PYTORCH_ROCM_ARCH"]
        except Exception as error:
            _failed = True
            warnings.warn(f"the HIP accumulation is not available ({type(error).__name__}: {error}); the Triton kernel is used", stacklevel=3)
    return _module


def accumulate_many(t, p, bins, quantized, weight, shared_weight=False):
    if not (shared_weight and quantized.shape[0] > 1 and bins <= MAXIMUM_BINS and available()):
        return None
    points_per_block = 8192 if bins == MAXIMUM_BINS else 4096
    if int(weight.min()) < 0 or int(weight.max()) * points_per_block >= 2**31:
        return None
    module = _load()
    if module is None:
        return None
    architecture = _device_architecture(t.device)
    if _built_for and architecture and architecture not in _built_for:  # a kernel built for other architectures is not launched
        if architecture not in _warned_architectures:
            _warned_architectures.add(architecture)
            warnings.warn(f"the HIP accumulation was built for {', '.join(sorted(_built_for))} and the device is {architecture}; the Triton kernel is used on it", stacklevel=2)
        return None
    fluxes, rows = quantized.shape[0], p.numel()
    sums = torch.zeros(fluxes, rows, bins, dtype=torch.int64, device=t.device)
    counts = torch.zeros(rows, bins, dtype=torch.int64, device=t.device)
    module.accumulate(t, (1.0 / p).contiguous(), quantized.contiguous(), weight[0].contiguous(), sums, counts, bins, points_per_block)
    return sums, counts.unsqueeze(0).expand(fluxes, -1, -1)
