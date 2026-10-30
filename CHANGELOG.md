# Changes

## Unreleased

### Added

- `bumps=True` on `box_search(...).power`, `search` and `search_many` scores the windows whose mean is above the whole's from the same fold, sums and grid as the dips. Each result then has a `bump` entry with the fields of a result, equal to the result of a search of the negated flux, and the results of `power` a `bump` attribute; the other fields are those of a search without `bumps`.

### Changed

- `search_many` without `boxes=True` keeps the fields of the best box (sum, start, width and number of points) at the best period of each flux and each cut and not at every trial period, as the SDE needs only the statistic of every period. A search of 10 fluxes over 8,060 periods holds 1.36 MB less at its peak (numpy allocations, `tracemalloc`; 4 of the 5 float64 or int64 arrays of fluxes by periods), and every field of every result is equal to that of the search with `boxes=True`, bit for bit.
- `search_many` with one flux and a matrix of weights, one row for each result, searches the flux once for each row from one fold of the times, and each result is that of a search of the flux with that row. A weight of zero now masks its point, which is dropped as a point with no flux is; before it was refused, and a negative or non-finite weight still is.
  A batch of five vectors takes 0.69 to 0.81 of the time of five separate searches (20,000 to 200,000 points) and not the time of one: each vector pays its own accumulation, 1,159 ms of the 1,531 ms of kernel time of five vectors at 100,000 points against 247 of 323 ms for one, because the weights differ and the one-pass kernels add the flux and the count of every point for every vector. The HIP kernel, which needs one set of weights, bounds a kernel that took them per vector at 0.40 of the batch's time at 100,000 points; it is not built, since a search with several sets of weights is rare next to the searches of many fluxes with one set, which share the fold already.
- The best boxes of a batch of periods are updated by one compiled function and not by about twenty eager passes: a search launches about half the kernels and takes 6% less time at 100,000 points; the numbers are the same to the bit.
- torch's limit of compiled variants of a function is raised to 64 when it is lower, and never lowered. At torch's default of 8, a process that used more combinations of options than that ran the scorer eagerly for the rest of its life, which changes the float32 statistic in its seventh digit.

### Fixed

- Exact mode (`transitgpu.tls`, backend `gpu`) returned wrong chi squares for part of the periods of a batch when the batch held more than 33,554,432 starts that passed the bounds of the screening. The correlations of those starts were launched at once, and a launch of that size is not dispatched whole; millions of them got other values, and the best period of a search could move. It needs about 524,000 such starts for each period at a batch of 64 periods, so a light curve with very many of them, or a batch of more than 256 periods chosen by the caller, could reach it. The checks of this package never reached it at the default batch. The correlations are now launched in groups of at most 8,388,608 starts, and a test launches 33,555,432 starts and compares them with a small launch.
- On CPU-only torch, importing the package's HIP helper no longer raises when it asks for the architecture of a device.

### Changed

- The exact search batches up to 256 periods, by the free device memory, instead of 64 for long light curves; the chi squares are the same as at 64. A batch whose mask of starts would pass 2^30 entries (periods times starts) is refused with an error that names both numbers.
