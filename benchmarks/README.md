# Benchmarks

`bench.py` times transitgpu against astropy's `BoxLeastSquares` and TLS 2.0.1; `results.json` has the runs of the README tables.

## Input

N evenly spaced points over 1,000 days (`--baseline`), flat flux with 1,000 ppm of Gaussian noise (seed 20260101) and one box transit (3.18 d, 3,000 ppm, 2.5 h), N = 20,000, 100,000 and 200,000 (`--sizes`); TLS's period grid from 0.5 to 14 days. `--npz FILE` times your light curve (arrays `time`, `flux`). `--gtls-input DAYS` makes the GTLS paper's timing input: DAYS days at 30-minute cadence, TLS's grid to half the span, rows `tls16-exact` and `tls-gpu`; `--t0-fit-margin` and `--noise-ppm` set `T0_fit_margin` and the noise. Each row is warmed up on 4,000 points, then timed over `--repeats` runs; the table gives the median and quartiles.

## Rows

| row | what it runs |
| --- | --- |
| `gpu` | the box search on the GPU |
| `cpu1`, `cpu16` | the same code on torch's CPU backend, 1 and 16 threads |
| `tls-gpu`, `tls-gpu-binned`, `tls-gpu-pl` | `transitgpu.tls` with the backend `gpu`, `gpu-binned` or `gpu-pl` |
| `hybrid` | `transitgpu.bls_tls_hybrid`: the box search, then the exact backend on a ±1% window around each of the 5 strongest peaks |
| `astropy` | `BoxLeastSquares`, 1 thread, objective `snr` |
| `tls1`, `tls16` | TLS 2.0.1, default backend `fused-pl`, 1 and 16 threads |
| `tls1-exact`, `tls16-exact` | TLS 2.0.1, exact backend `fused`, 1 and 16 threads |
| `many`, `tls16-many` | 64 light curves on one time axis: one `search_many` call, and TLS (`fused-pl`, 16 threads) one curve at a time; only run when named, with `--sizes 20000` |
| `gpu-shuffles`, `gpu-bumps`, `gpu-two`, `gpu-weights`, `gpu-weights-separate` | shuffled copies; dips and bumps in 1 or 2 calls; 5 weight vectors in 1 or 5; named only |

## Timed region

| row | inside the timer | outside the timer |
| --- | --- | --- |
| `gpu`, `cpu1`, `cpu16` | cleaning and quantising the flux, the copy to the GPU (`gpu` only), folding and scoring, copying the best boxes back, the SDE and the period uncertainty | the period grid (passed in), compilation |
| `astropy` | `power(periods, durations, "snr")` | building the object; no grid, SDE or fit |
| `tls-gpu`, `tls-gpu-binned`, `tls-gpu-pl`, `tls1`, `tls16`, `tls1-exact`, `tls16-exact` | the whole `power()` call: cleaning, both grids, the template cache, the period search (GPU transfers included), the SDE and the final fit | compilation |
| `many`, `tls16-many` | the whole set: one `search_many` call, or 64 `power()` calls | building the set, compilation |
| `hybrid` | the box search with its grid, then `power()` on each of the 5 windows | compilation |

## Conditions

`session_note` of `results.json` names the session and its conditions.

## Commands

```
taskset -c 0-15 uv run --group rocm python benchmarks/bench.py --rows gpu hybrid tls-gpu tls-gpu-binned astropy tls1 tls16 tls1-exact tls16-exact --repeats 7 --max-run-seconds 600 --json bench.json
taskset -c 0-15 uv run --group rocm python benchmarks/bench.py --sizes 20000 --rows many tls16-many --repeats 7 --json bench-many.json
uv run --group rocm python benchmarks/bench.py --readme-table bench.json
uv run --group rocm python benchmarks/peak.py
```
