# Changes

## Unreleased

### Fixed

- Exact mode (`transitgpu.tls`, backend `gpu`) returned wrong chi squares for part of the periods of a batch when the batch held more than 33,554,432 starts that passed the bounds of the screening. The correlations of those starts were launched at once, and a launch of that size is not dispatched whole; millions of them got other values, and the best period of a search could move. It needs about 524,000 such starts for each period at a batch of 64 periods, so a light curve with very many of them, or a batch of more than 256 periods chosen by the caller, could reach it. The checks of this package never reached it at the default batch. The correlations are now launched in groups of at most 8,388,608 starts, and a test launches 33,555,432 starts and compares them with a small launch.
- On CPU-only torch, importing the package's HIP helper no longer raises when it asks for the architecture of a device.

### Changed

- The exact search batches up to 256 periods, by the free device memory, instead of 64 for long light curves; the chi squares are the same as at 64. A batch whose mask of starts would pass 2^30 entries (periods times starts) is refused with an error that names both numbers.
