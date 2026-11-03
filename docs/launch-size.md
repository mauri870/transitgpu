# Launch size of the box search on long stacks

The one-pass accumulation (`accumulate_many`, a Triton or HIP kernel) folds every point of a stack into the phase bins of a batch of trial periods with integer atomic additions. One launch visits every point once for every period of the batch, so its duration is the number of periods times the number of points times the cost of one visit.

`search_many` takes `ONE_PASS_BATCH = 512` periods at a time whatever the number of points. The duration of a launch therefore grows in proportion to the points: the largest launch of the main kernel on a RX 7900 XTX was 1.16 ms at 50,000 points, 5.24 ms at 250,000 and 23.07 ms at 1,000,000 (`benchmarks/launch_size.py`), about 45 ps per visit, and a stack of 2.5 million points (two years at 25 s) would make launches of about 58 ms. A GPU that also drives a display wants launches of about 10 ms; a launch that runs much longer starves the display and, past the driver's timeout, resets the device.

## Change

The periods per launch become `min(ONE_PASS_BATCH, ONE_PASS_ELEMENTS // points)`, at least 1, with `ONE_PASS_ELEMENTS = 256_000_000` (periods x points per launch, about 11.5 ms at the rate above). A stack of up to 500,000 points keeps 512 periods a launch, as today; larger stacks take fewer periods a launch, so that no launch visits more than 256 million (period, point) pairs. `batch=` given by the caller still overrides.

Why not a slice of the points as well. The sums are integers added atomically, so the order of the additions does not change the result, and a period is independent of the others: the periods of a launch can be any subset of the batch and the result is bitwise the same. A single period of 2.5 million points is one launch of about 0.11 ms, so slicing the points would only matter above about 90 million points (a launch of one period over 10 ms), which no stack of this kind has. Points are not sliced; the limit is documented, and the test that a one-period launch is short is the benchmark above.

## Tests

1. `launch_rows` (the rule) at the boundaries: 512 up to 500,000 points, the quotient above, 1 at the largest sizes, and `batch=` overriding.
2. The search of a small stack with the budget made so small that the batch is 3 periods, against the default batch of 512: the same result, bit for bit (device test).
3. Stacks of the size of a TESS stack (up to 500,000 points) take 512 periods a launch: the same value as before the change.
4. `benchmarks/launch_size.py`: the largest launch of the main kernel at 50,000, 250,000, 1,000,000, 2,000,000 and 2,525,000 points, each size only after the smaller ones, each under a time limit; the kernel journal is read before and after by the person who runs it.
