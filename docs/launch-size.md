# Launch size of the box search on long stacks

The one-pass accumulation (`accumulate_many`, a Triton or HIP kernel) folds every point of a stack into the phase bins of a batch of trial periods with integer atomic additions. One launch visits every point once for every period of the batch, so its duration is the number of periods times the number of points times the cost of one visit.

With 512 periods a launch whatever the number of points, the duration of a launch grows in proportion to the points. On a RX 7900 XTX the largest launch was 5.25 ms at 312,500 points, 8.46 ms at 450,000 and 19.81 ms at 1,000,000 (`benchmarks/launch_compare.py`), and a stack of 2.5 million points (two years at 25 s) would make launches of about 50 ms by the same proportion (19.81 ms at 1,000,000 points times 2.525). A GPU that also drives a display wants launches of about 10 ms; a launch that runs much longer starves the display and, past the driver's timeout, resets the device.

## Change

The periods per launch are `core.one_pass_batch(points) = max(1, min(ONE_PASS_BATCH, ONE_PASS_ELEMENTS // points))`, with `ONE_PASS_BATCH = 512` and `ONE_PASS_ELEMENTS = 160_000_000` (period and point pairs a launch, 4.2 ms for the accumulation kernel on that card, 4.3 ms at 312,500 points and 4.2 ms at 2,525,000). A stack of up to 312,500 points keeps 512 periods a launch; a larger one takes fewer. `batch=` given by the caller still overrides.

The sums are integers added atomically, so the order of the additions does not change the result, and a period is independent of the others: the periods of a launch can be any subset of the batch and the result is bitwise the same. One period costs about 0.07 ms at 2.5 million points, so a single period would pass 10 ms only above about 380 million points. Points are not sliced.

## Measured

The table of the README gives the periods a launch, the largest launch of any kernel and the search time at 312,500 to 2,525,000 points. The largest launch is 6.5 ms at 1,000,000 points against 19.8 ms with 512 periods, and 9.0 ms at 2,525,000 points. The search takes 1.3 to 5.5 % longer with the bound at 350,000 to 1,000,000 points, where the launches are more numerous and each launch has a fixed cost.

## Tests

`tests/test_launch_rows.py`:

1. The rule at its boundaries: 512 up to 312,500 points, the quotient above, 1 at the largest sizes, and `batch=` overriding.
2. A search with a batch of 3 periods gives the same result as with 512, bit for bit (device).
3. The mean launch of the longest kernel at 2,525,000 points is at most 1.5 times its mean at 312,500 points, and no launch at 2,525,000 points exceeds 10 ms (device). The largest launch is not compared between sizes: it is the maximum of 18 launches at one size and of about 1,000 at the other.

`benchmarks/launch_size.py` prints the launches of every kernel at each size, and `benchmarks/launch_compare.py` the table. Read the kernel log (`journalctl -k`) before and after.
