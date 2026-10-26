"""Write paper/numbers.tex from benchmarks/results.json: a macro per timed row and size, the machine and the versions.

The paper reads a time as \\T{row}{points}, for example \\T{gpu}{200000}, and the machine as \\Hw{cpu}. A time that
results.json does not hold prints as a bracketed name, so a draft builds before every benchmark has run. Where two
blocks of results.json hold the same row at the same size, the block named first in BLOCKS is used (None is the top
of the file).

    python paper/scripts/fill_numbers.py [--results benchmarks/results.json] [--out paper/numbers.tex]
"""

import argparse
import json
import sys
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # completeness.py, for the Wilson interval and the merge of the real stars
from completeness import merge_real  # noqa: E402

PACKAGE = HERE.parent.parent
BLOCKS = (None, "tls", "tls_earlier_session")  # None: the sizes at the top of the file
SUFFIXED = ("gtls_benchmark_sizes", "block_shuffles")


def seconds(value):
    """Return a time with three significant figures, as the paper prints it."""
    if value >= 100:
        return f"{value:.0f}"
    rounded = float(f"{value:.3g}")
    return f"{rounded:.{max(0, 2 - math.floor(math.log10(abs(rounded))))}f}"  # keep trailing zeros: 12.0, not 12


def rows_of(block, suffix=""):
    """Return (row + suffix, points, median seconds, trial periods) for every row of a block that has `sizes`."""
    found = []
    for points, size in block.get("sizes", {}).items():
        for row, measured in size["rows"].items():
            found.append((row + suffix, int(points), measured["median"], measured["trial_periods"]))
    return found


def collect(results):
    """Return {(row, points): (median seconds, trial periods)} from results.json.

    The blocks of BLOCKS come first, in order, then the runs of other baselines and the shuffles, which have rows of
    their own.
    """
    table = {}
    for name in BLOCKS:
        for row, points, median, periods in rows_of(results if name is None else results.get(name, {})):
            table.setdefault((row, points), (median, periods))
    for run in results.get("gtls_benchmark_sizes", {}).get("runs", []):
        for row, points, median, periods in rows_of(run):
            table.setdefault((row, points), (median, periods))
    for run in results.get("gtls_input", {}).get("runs", []):
        for row, points, median, periods in rows_of(run, "" if run["t0_fit_margin"] == results["t0_fit_margin"] else "-margin"):  # the runs at GTLS's T0 margin keep their rows under another name
            table.setdefault((row, points), (median, periods))
    for block in results.get("block_shuffles", {}).values():
        if isinstance(block, dict) and "sizes" in block:
            for row, points, median, periods in rows_of(block):
                table.setdefault((row, points), (median, periods))
    return table


# ratios of the paper: (slower row, faster row), read as the time of the first over the time of the second at one size
RATIOS = (("tls16", "gpu"), ("tls16-exact", "tls-gpu"), ("tls16", "tls-gpu-binned"), ("tls1", "gpu"), ("tls1-exact", "tls-gpu"), ("astropy", "gpu"), ("tls-gpu", "gpu"), ("tls-gpu", "tls-gpu-binned"),
          ("tls16", "hybrid"), ("tls16-many", "many"), ("tls16-exact-margin", "tls-gpu-margin"), ("tls16-exact", "tls-gpu-binned"))


def ratio(value):
    """Return a ratio to two significant figures (an integer from 100 up)."""
    return f"{value:.2g}" if value < 100 else f"{value:.0f}"


# the times that GTLS reports for one RTX 4090 (GTLS) and for a Ryzen 9 7950X (TLS), by the points of the light curve (1,500 and 3,000 days at 48 points a day):
# Hu et al. 2026, Sect. 3.4
GTLS_PUBLISHED = {72000: {"time": 33.3, "tls": 522}, 144000: {"time": 138, "tls": 3289}}


def gtls_macros(results):
    """The macros \\Gt{time}{N} and \\Gt{tls}{N} (the published seconds), \\Gt{ratio}{N} (their GTLS time over our exact mode at the margin of GTLS) and
    \\Gt{tlsratio}{N} (their TLS time over our TLS at that margin), both to two significant figures and absent when the row of the margin is missing."""
    table = collect(results)
    lines = ["\\providecommand{\\Gt}[2]{\\ifcsname gt@#1@#2\\endcsname\\csname gt@#1@#2\\endcsname\\else[\\texttt{#1,#2}]\\fi}"]
    for points, published in GTLS_PUBLISHED.items():
        values = {name: format(published[name], ",g").replace(",", "\\,") for name in ("time", "tls")}  # a number of four digits has a thin space, as everywhere in the paper
        if ("tls-gpu-margin", points) in table:
            values["ratio"] = ratio(published["time"] / table[("tls-gpu-margin", points)][0])
        if ("tls16-exact-margin", points) in table:
            values["tlsratio"] = ratio(published["tls"] / table[("tls16-exact-margin", points)][0])
        lines += [f"\\expandafter\\def\\csname gt@{name}@{points}\\endcsname{{{text}}}" for name, text in values.items()]
    return "\n".join(lines) + "\n"


def macros(results):
    lines = []
    for (row, points), (median, periods) in sorted(collect(results).items()):
        lines.append(f"\\expandafter\\def\\csname t@{row}@{points}\\endcsname{{{seconds(median)}}}")
        lines.append(f"\\expandafter\\def\\csname p@{row}@{points}\\endcsname{{{periods:,}}}".replace(",", "\\,"))
    table = collect(results)
    for slow, fast in RATIOS:
        for (row, points), (median, _) in sorted(table.items()):
            if row == fast and (slow, points) in table:
                lines.append(f"\\expandafter\\def\\csname r@{slow}@{fast}@{points}\\endcsname{{{ratio(table[(slow, points)][0] / median)}}}")
    for key, value in results.get("hardware", {}).items():
        lines.append(f"\\expandafter\\def\\csname h@{key}\\endcsname{{{value}}}")
    for key, value in results.get("versions", {}).items():
        lines.append(f"\\expandafter\\def\\csname v@{key}\\endcsname{{{value}}}".replace("_", "\\_"))
    return "\n".join(lines) + "\n"


def threshold_macros(analysis):
    """The macros of the shuffle calibration, from the analysis file of validation/threshold_analysis.py: \\th{class}{field} for the curves with nothing injected (count of curves above their threshold, curves, share and its interval in percent) and \\thi{class}{field} for the
    inverted-light-curve null. Empty without an analysis."""
    lines = []
    for prefix, block in (("th", analysis.get("null", {})), ("thi", analysis.get("inverted", {}))):
        for name, entry in block.items():
            for field, text in (("above", f"{entry['count']}"), ("of", f"{entry['of']}"), ("share", f"{100 * entry['share']:.1f}"), ("low", f"{100 * entry['low']:.1f}"), ("high", f"{100 * entry['high']:.1f}")):
                lines.append(f"\\expandafter\\def\\csname {prefix}@{name}@{field}\\endcsname{{{text}}}")
    rotation = analysis.get("variable_rotation")
    if rotation:
        from completeness import wilson

        explained, rest = rotation["crossings_explained"], rotation["not_crossing_explained"]
        unexplained = rotation["crossings_not_explained"]
        low, high = wilson(unexplained, rotation["curves"])
        for name, text in (("explained@count", f"{explained['count']:,}".replace(",", "\\,")), ("explained@of", f"{explained['of']:,}".replace(",", "\\,")), ("explained@share", f"{100 * explained['share']:.1f}"), ("explained@low", f"{100 * explained['low']:.1f}"),
                           ("explained@high", f"{100 * explained['high']:.1f}"), ("unexplained@count", f"{unexplained}"), ("unexplained@share", f"{100 * unexplained / rotation['curves']:.2f}"), ("unexplained@low", f"{100 * low:.1f}"), ("unexplained@high", f"{100 * high:.1f}"),
                           ("rest@count", f"{rest['count']}"), ("rest@of", f"{rest['of']}"), ("rest@share", f"{100 * rest['share']:.1f}")):
            lines.append(f"\\expandafter\\def\\csname vr@{name}\\endcsname{{{text}}}")
    lines += real_threshold_macros(analysis.get("real"))
    for name, key in (("median", "0.5"), ("p90", "0.9"), ("p99", "0.99")):
        value = analysis.get("false_alarm_quantiles_of_a_threshold_of_48", {}).get(key)
        if value is not None:
            lines.append(f"\\expandafter\\def\\csname beta@{name}\\endcsname{{{100 * value:.1f}}}")
    return "\n".join(lines) + "\n" if lines else ""


def real_threshold_macros(real):
    """The macros \\Thr{n}, \\Thr{mean}, \\Thr{ks} and \\Thr{p} for the stars of the current list in the shuffle calibration on real stars (their number, the mean share of the rank of the best peak among its shuffles to two decimals, the Kolmogorov distance to
    the uniform distribution to two decimals and its p-value to two significant figures) and the same for the pooled stars as \\Thr{pooledn}, \\Thr{pooledmean}, \\Thr{pooledks} and \\Thr{pooledp}. No lines without the real part."""
    if not real:
        return []
    lines = ["\\providecommand{\\Thr}[1]{\\ifcsname thr@#1\\endcsname\\csname thr@#1\\endcsname\\else[\\texttt{#1}]\\fi}"]
    for prefix, key in (("", "current_list"), ("pooled", "pooled")):
        block = real.get(key)
        if block:
            for name, text in (("n", f"{block['stars']}"), ("mean", f"{block['mean_rank_share']:.2f}"), ("ks", f"{block['kolmogorov']:.2f}"), ("p", f"{block['p']:.2g}")):
                lines.append(f"\\expandafter\\def\\csname thr@{prefix}{name}\\endcsname{{{text}}}")
    return lines


def one_decimal(value):
    """A size in points to one decimal, halves rounded up (1.375 is 1.4), without a trailing zero."""
    from decimal import ROUND_HALF_UP, Decimal

    text = str(Decimal(f"{value:.6f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return text[:-2] if text.endswith(".0") else text


def recovery_macros(results, rate="0.05"):
    """The macros of the paired recovery differences, from the results.json of validation/analyse.py: \\pd@class@search@field for the difference of the share of injected planets found by a search from that of exact mode at the false-alarm rate `rate`,
    in points (est, low and high are signed, size is the absolute value to one decimal, round to an integer), and pd@search@maxsize for the largest size over the classes, to one decimal with halves rounded up. Empty without a paired block."""
    lines, largest = [], {}

    def signed(value):
        text = f"{100 * value:+.1f}"
        return "$" + ("0.0" if text in ("+0.0", "-0.0") else text) + "$"

    for part, content in results.items():
        pairs = (content.get("paired") or {}).get(rate) or {}
        name = part.removeprefix("main-")
        for key, entry in pairs.items():
            search = key.removesuffix("_minus_exact")
            size = abs(100 * entry["estimate"])
            largest[search] = max(largest.get(search, 0.0), size)
            for field, text in (("est", signed(entry["estimate"])), ("low", signed(entry["low"])), ("high", signed(entry["high"])), ("size", f"{size:.1f}"), ("round", f"{round(size)}")):
                lines.append(f"\\expandafter\\def\\csname pd@{name}@{search}@{field}\\endcsname{{{text}}}")
    steps = 0.0
    for content in results.values():
        pairs = (content.get("paired") or {}).get(rate) or {}
        if "hybrid20_minus_exact" in pairs and "hybrid5_minus_exact" in pairs:
            steps = max(steps, abs(100 * (pairs["hybrid20_minus_exact"]["estimate"] - pairs["hybrid5_minus_exact"]["estimate"])))
    if steps:
        lines.append(f"\\expandafter\\def\\csname pd@hybrid20from5@maxsize\\endcsname{{{one_decimal(steps)}}}")
    for search, size in sorted(largest.items()):
        lines.append(f"\\expandafter\\def\\csname pd@{search}@maxsize\\endcsname{{{one_decimal(size)}}}")
    lines += real_share_macros(results.get("real"), rate)
    return "\n".join(lines) + "\n" if lines else ""


def real_share_macros(real, rate="0.05"):
    """The macros of the share of the injected planets that a search finds in the light curves of real stars at a target signal to noise of 6, 8 and 10: \\Rr{search}{snr} in percent as an integer and \\Rr{search}{snrlow} and
    \\Rr{search}{snrhigh} for its Wilson bounds, \\Rr{n}{planets} for the planets in a cell and \\Rr{injected}{planets} for those injected in all. Empty without the real part."""
    if not real:
        return []
    lines = ["\\providecommand{\\Rr}[2]{\\ifcsname rr@#1@#2\\endcsname\\csname rr@#1@#2\\endcsname\\else[\\texttt{#1, #2}]\\fi}",
             f"\\expandafter\\def\\csname rr@injected@planets\\endcsname{{{real['injected']}}}"]
    cells = {}
    for search, entry in real["searches"].items():
        for level, cell in (entry.get(rate) or {}).get("by_snr", {}).items():
            if float(level) in (6, 8, 10):
                cells[(search, f"{float(level):g}")] = cell
    for (search, level), cell in sorted(cells.items()):
        low, high = cell["wilson"]
        for field, value in (("", cell["found"] / cell["n"]), ("low", low), ("high", high)):
            lines.append(f"\\expandafter\\def\\csname rr@{search}@{level}{field}\\endcsname{{{100 * value:.0f}}}")
    if cells:
        lines.append(f"\\expandafter\\def\\csname rr@n@planets\\endcsname{{{next(iter(cells.values()))['n']}}}")
    return lines


def box_variable_macros(box, rate="0.05"):
    """The macros of the three box searches and exact mode on the injections into the variable class, from paper/box_variable.json: \\Bv{search}{found} (planets found of those injected), {share}, {low} and {high} (percent, one decimal, the Wilson interval)
    and {threshold} (the SDE at the false-alarm rate `rate`, two decimals) for astropy, box, box_detrended (also as detrended) and exact; \\Bvd{pair}{est}, {low} and {high} for the paired differences in points (signed, one decimal) under the names that the file gives
    (box_minus_exact) and under the short ones (box-exact). Empty without a file."""
    if not box:
        return ""
    lines = ["\\providecommand{\\Bv}[2]{\\ifcsname bv@#1@#2\\endcsname\\csname bv@#1@#2\\endcsname\\else[\\texttt{#1, #2}]\\fi}",
             "\\providecommand{\\Bvd}[2]{\\ifcsname bvd@#1@#2\\endcsname\\csname bvd@#1@#2\\endcsname\\else[\\texttt{#1, #2}]\\fi}"]

    def signed(value):
        text = f"{100 * value:+.1f}"
        return "$" + ("0.0" if text in ("+0.0", "-0.0") else text) + "$"

    for search, entry in box["searches"].items():
        cell = entry[rate]
        values = (("found", f"{cell['found']}"), ("share", f"{100 * cell['found'] / cell['n']:.1f}"), ("low", f"{100 * cell['wilson'][0]:.1f}"), ("high", f"{100 * cell['wilson'][1]:.1f}"), ("threshold", f"{cell['threshold']:.2f}"))
        for name in dict.fromkeys((search, "detrended" if search == "box_detrended" else search)):
            lines += [f"\\expandafter\\def\\csname bv@{name}@{field}\\endcsname{{{text}}}" for field, text in values]
    for pair, entry in box["paired"][rate].items():
        first, second = pair.split("_minus_")
        short = f"{first.removeprefix('box_') if first == 'box_detrended' else first}-{second.removeprefix('box_') if second == 'box_detrended' else second}"
        for name in (pair, short):
            for field, key in (("est", "estimate"), ("low", "low"), ("high", "high")):
                lines.append(f"\\expandafter\\def\\csname bvd@{name}@{field}\\endcsname{{{signed(entry[key])}}}")
    return "\n".join(lines) + "\n"


def split_macros(profile):
    """The macros of the split of the device time by stage, from paper/scripts/profile_split.json: sp@points@stage for the seconds of fold, scans, screening, correlations and rest, and sp@points@total, to two decimals. Empty without a profile."""
    lines = []
    for entry in profile:
        for stage, seconds in (*entry["stages"].items(), ("total", entry["device_seconds"])):
            lines.append(f"\\expandafter\\def\\csname sp@{entry['points']}@{stage}\\endcsname{{{seconds:.2f}}}")
    return "\n".join(lines) + "\n" if lines else ""


def cold_macros(results):
    """The macros \\Cold{row}{kept} and \\Cold{row}{cleared}: the seconds of the first call of a fresh process for the row, with the compiled caches kept and with the caches cleared,
    from the block cold_first_call of results.json; a row missing from a dictionary has no macro. Empty without the block."""
    cold = results.get("cold_first_call")
    if not cold:
        return ""
    lines = ["\\providecommand{\\Cold}[2]{\\ifcsname cold@#1@#2\\endcsname\\csname cold@#1@#2\\endcsname\\else[\\texttt{#1,#2}]\\fi}"]
    for kind, key in (("kept", "caches_kept"), ("cleared", "caches_cleared")):
        lines += [f"\\expandafter\\def\\csname cold@{row}@{kind}\\endcsname{{{seconds(value)}}}" for row, value in cold.get(key, {}).items()]
    return "\n".join(lines) + "\n"


def adversarial_macros(cases):
    """The macros \\Adv{case}{field} for the adversarial cases of paper/adversarial_table.json: chi2 and other (the largest relative differences, two significant figures, infinity where a field has none),
    periodtls and periodgpu (the best periods, four decimals); \\Adv{all}{chi2} is the largest chi2 over every case and \\Adv{all}{other} the largest other over the cases whose best period is the same.
    Empty without cases."""
    if not cases:
        return ""

    def power(value):
        if value == float("inf"):
            return "\\ensuremath{\\infty}"
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"\\ensuremath{{{mantissa}\\times10^{{{int(exponent)}}}}}"

    lines = ["\\providecommand{\\Adv}[2]{\\ifcsname adv@#1@#2\\endcsname\\csname adv@#1@#2\\endcsname\\else[\\texttt{#1,#2}]\\fi}"]
    for name, case in cases.items():
        values = {"chi2": power(case["chi2_max_relative"]), "other": power(case["other_fields_max_relative"]), "periodtls": f"{case['best_period_tls']:.4f}", "periodgpu": f"{case['best_period_gpu']:.4f}"}
        lines += [f"\\expandafter\\def\\csname adv@{name}@{field}\\endcsname{{{text}}}" for field, text in values.items()]
    maxima = {"chi2": power(max(case["chi2_max_relative"] for case in cases.values())), "other": power(max(case["other_fields_max_relative"] for case in cases.values() if case["best_period_same"]))}
    lines += [f"\\expandafter\\def\\csname adv@all@{field}\\endcsname{{{text}}}" for field, text in maxima.items()]
    return "\n".join(lines) + "\n"


def roc_macros(roc):
    """The macros \\Roc{part}{search}{rate} (the share of the injected light curves found, in percent to one decimal, at a false-alarm rate of 0.01 or 0.05 of the part's nulls), \\Rocl and \\Roch (the lower and upper bound of its Wilson
    interval) from the roc.json of validation/roc.py. The part loses the prefix "main-". Empty without a file."""
    if not roc:
        return ""
    lines = [f"\\providecommand{{\\{name}}}[3]{{\\ifcsname {key}@#1@#2@#3\\endcsname\\csname {key}@#1@#2@#3\\endcsname\\else[\\texttt{{#1,#2,#3}}]\\fi}}" for name, key in (("Roc", "roc"), ("Rocl", "rocl"), ("Roch", "roch"))]
    for part, content in roc["parts"].items():
        name = part.removeprefix("main-")
        for search, entry in content["searches"].items():
            for rate in (0.01, 0.05):
                index = roc["fpr"].index(rate)
                for field, value in (("roc", entry["tpr"][index]), ("rocl", entry["low"][index]), ("roch", entry["high"][index])):
                    lines.append(f"\\expandafter\\def\\csname {field}@{name}@{search}@{rate:g}\\endcsname{{{100 * value:.1f}}}")
    return "\n".join(lines) + "\n"


def period_macros(results, rate="0.05"):
    """The macros \\Rp{part}{search}{period} (the share found, in percent to one decimal), \\Rpl and \\Rph (the lower and upper bound of its Wilson interval) from the by_period cells of the recovery results at the false-alarm
    rate: every injected light curve of that period at every signal to noise of the grid, as the file holds them. The part loses the prefix "main-". Empty without results."""
    if not results:
        return ""
    lines = ["\\providecommand{\\Rp}[3]{\\ifcsname rp@#1@#2@#3\\endcsname\\csname rp@#1@#2@#3\\endcsname\\else[\\texttt{#1,#2,#3}]\\fi}",
             "\\providecommand{\\Rpl}[3]{\\ifcsname rpl@#1@#2@#3\\endcsname\\csname rpl@#1@#2@#3\\endcsname\\else[\\texttt{#1,#2,#3}]\\fi}",
             "\\providecommand{\\Rph}[3]{\\ifcsname rph@#1@#2@#3\\endcsname\\csname rph@#1@#2@#3\\endcsname\\else[\\texttt{#1,#2,#3}]\\fi}"]
    for part, content in results.items():
        name = part.removeprefix("main-")
        for search, entry in content["searches"].items():
            for period, cell in entry.get(rate, {}).get("by_period", {}).items():
                for field, value in (("rp", 100 * cell["found"] / cell["n"]), ("rpl", 100 * cell["wilson"][0]), ("rph", 100 * cell["wilson"][1])):
                    lines.append(f"\\expandafter\\def\\csname {field}@{name}@{search}@{period}\\endcsname{{{value:.1f}}}")
    return "\n".join(lines) + "\n"


def peak_macros(peak):
    """The macros of the attained throughput of the GPU, from the paper/peak.json that benchmarks/peak.py writes: \\Pk{fp32} and \\Pk{fp64} in TFLOPS (to three significant figures), \\Pk{ratio} (attained FP32 over FP64, an integer), \\Pk{copy} in GB/s (an integer),
    \\Pk{copyshare} in percent of the specified bandwidth (when the file has it) and \\Pk{clockmax} in MHz. A run beside other processes on the device is named in a comment. Empty without a file."""
    if not peak:
        return ""
    lines = ["\\providecommand{\\Pk}[1]{\\ifcsname pk@#1\\endcsname\\csname pk@#1\\endcsname\\else[\\texttt{#1}]\\fi}"]
    if peak.get("other_processes_on_the_device"):
        lines.append("% other processes held the device: " + ", ".join(str(pid) for pid in peak["other_processes_on_the_device"]))
    values = {"fp32": f"{peak['fp32_tflops']:.3g}", "fp64": f"{peak['fp64_tflops']:.3g}", "ratio": f"{peak['fp32_over_fp64']:.0f}", "copy": f"{peak['copy_gb_per_s']:.0f}", "clockmax": f"{peak['clock_mhz']['max']:.0f}"}
    if "copy_share_of_spec" in peak:
        values["copyshare"] = f"{100 * peak['copy_share_of_spec']:.0f}"
    lines += [f"\\expandafter\\def\\csname pk@{name}\\endcsname{{{text}}}" for name, text in values.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=PACKAGE / "benchmarks" / "results.json")
    parser.add_argument("--out", type=Path, default=HERE.parent / "numbers.tex")
    parser.add_argument("--threshold", type=Path, default=HERE.parent / "threshold.json", help="the analysis file of the shuffle calibration, when the paper has one")
    parser.add_argument("--recovery", type=Path, default=HERE.parent / "recovery.json", help="the results.json of the recovery of injected planets, when the paper has one")
    parser.add_argument("--profile", type=Path, default=HERE / "profile_split.json")
    parser.add_argument("--profile-out", type=Path, default=HERE.parent / "split_numbers.tex")
    parser.add_argument("--peak", type=Path, default=HERE.parent / "peak.json", help="the results of benchmarks/peak.py, when the paper has them")
    parser.add_argument("--peak-out", type=Path, default=HERE.parent / "peak_numbers.tex")
    parser.add_argument("--adversarial", type=Path, default=HERE.parent / "adversarial_table.json", help="the maxima of the adversarial cases, from scripts/adversarial.py")
    parser.add_argument("--adversarial-out", type=Path, default=HERE.parent / "adversarial_numbers.tex")
    parser.add_argument("--recovery-real", type=Path, default=HERE.parent / "recovery_real.json", help="the results of the injections into real stars, when the paper has them")
    parser.add_argument("--box-variable", type=Path, default=HERE.parent / "box_variable.json", help="the results of the box searches on the variable class, when the paper has them")
    parser.add_argument("--roc", type=Path, default=HERE.parent / "roc.json", help="the ROC curves of the searches, from validation/roc.py, when the paper has them")
    parser.add_argument("--roc-out", type=Path, default=HERE.parent / "roc_numbers.tex")
    parser.add_argument("--recovery-out", type=Path, default=HERE.parent / "recovery_numbers.tex")
    parser.add_argument("--threshold-out", type=Path, default=HERE.parent / "threshold_numbers.tex")
    arguments = parser.parse_args()
    arguments.out.write_text("% written by scripts/fill_numbers.py from benchmarks/results.json\n" + (lambda results: macros(results) + gtls_macros(results) + cold_macros(results))(json.loads(arguments.results.read_text())))
    if arguments.profile.exists():
        arguments.profile_out.write_text("% written by scripts/fill_numbers.py from profile_split.json\n" + split_macros(json.loads(arguments.profile.read_text())))
    if arguments.peak.exists():
        arguments.peak_out.write_text("% written by scripts/fill_numbers.py from peak.json\n" + peak_macros(json.loads(arguments.peak.read_text())))
    if arguments.adversarial.exists():
        arguments.adversarial_out.write_text("% written by scripts/fill_numbers.py from adversarial_table.json\n" + adversarial_macros(json.loads(arguments.adversarial.read_text())))
    if arguments.recovery.exists():
        recovery = json.loads(arguments.recovery.read_text())
        if arguments.recovery_real.exists():
            recovery = merge_real(recovery, json.loads(arguments.recovery_real.read_text()))
        arguments.recovery_out.write_text("% written by scripts/fill_numbers.py from recovery.json, recovery_real.json and box_variable.json\n" + recovery_macros(recovery) + period_macros(recovery)
                                            + (box_variable_macros(json.loads(arguments.box_variable.read_text())) if arguments.box_variable.exists() else ""))
    if arguments.roc.exists():
        arguments.roc_out.write_text("% written by scripts/fill_numbers.py from roc.json\n" + roc_macros(json.loads(arguments.roc.read_text())))
    if arguments.threshold.exists():
        arguments.threshold_out.write_text("% written by scripts/fill_numbers.py from threshold.json\n" + threshold_macros(json.loads(arguments.threshold.read_text())))


if __name__ == "__main__":
    main()
