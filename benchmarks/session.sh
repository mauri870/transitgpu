#!/bin/bash
# One timing session for the paper: every part runs in this one script, pinned to cores 0-15, on a machine with nothing else on the device (bench.py --refuse-if-busy stops a row if another process holds it).
# The load average and the free memory are written before and after each part to DIRECTORY/session.jsonl. Then: uv run python benchmarks/assemble.py DIRECTORY
#   bash benchmarks/session.sh DIRECTORY
# Stop it by touching DIRECTORY/STOP; a part that has its file is skipped on a restart.
set -u
directory=${1:?directory}
mkdir -p "$directory"
cd "$(dirname "$0")/.."
bench="taskset -c 0-15 uv run python benchmarks/bench.py --refuse-if-busy"

log() {  # log PART start|end
  python3 - "$directory" "$1" "$2" <<'PY'
import json, os, sys, time
directory, part, edge = sys.argv[1:]
free = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable"))
with open(os.path.join(directory, "session.jsonl"), "a") as handle:
    handle.write(json.dumps({"part": part, "edge": edge, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load": os.getloadavg()[0], "memory_available_kib": free}) + "\n")
PY
}

part() {  # part NAME COMMAND...
  name=$1; shift
  [ -e "$directory/STOP" ] && { echo "stopped before $name"; exit 0; }
  [ -e "$directory/$name.json" ] && { echo "$name: done already"; return; }
  log "$name" start
  "$@" --json "$directory/$name.json"
  log "$name" end
}

# A. the first call of a fresh process, one row each (the row many is the one that uses the HIP kernel, which hipcc builds at its first call): with the compiled caches kept, then with them cleared (empty directories for the caches of Triton and of torch's extensions, so that the caches of the user are not touched)
for row in gpu hybrid tls-gpu tls-gpu-binned tls-gpu-pl many; do
  part "cold-$row" $bench --sizes 20000 --rows "$row" --repeats 1
done
for row in gpu tls-gpu many; do
  cleared=$(mktemp -d)
  part "cold-cleared-$row" env TRITON_CACHE_DIR="$cleared/triton" TORCH_EXTENSIONS_DIR="$cleared/torch" $bench --sizes 20000 --rows "$row" --repeats 1
done
# B. the table of the paper: 7 calls at the three sizes
part main $bench --rows gpu hybrid tls-gpu tls-gpu-binned tls-gpu-pl tls16 tls16-exact --repeats 7
# D. the other rows of the paper
part many $bench --sizes 20000 --rows many --repeats 7
part many-cpu $bench --sizes 20000 --rows tls16-many --repeats 7
part shuffles $bench --rows gpu-shuffles --repeats 7
part gtls72 $bench --sizes 72000 --baseline 1500 --period-max 750 --rows gpu --repeats 7
part gtls144 $bench --sizes 144000 --baseline 3000 --period-max 1500 --rows gpu --repeats 7
# the input of the GTLS paper's timing: TLS 2.0.1 on 16 threads, exact mode, binned mode and the box search at TLS's T0 margin, and the two exact rows at the margin of GTLS (0.125)
part gtls-input-1500 $bench --gtls-input 1500 --rows tls16-exact tls-gpu tls-gpu-binned gpu --repeats 7 --max-run-seconds 7200
part gtls-input-3000 $bench --gtls-input 3000 --rows tls16-exact tls-gpu tls-gpu-binned gpu --repeats 7 --max-run-seconds 7200
part gtls-input-1500-margin $bench --gtls-input 1500 --t0-fit-margin 0.125 --rows tls16-exact tls-gpu --repeats 7 --max-run-seconds 7200
part gtls-input-3000-margin $bench --gtls-input 3000 --t0-fit-margin 0.125 --rows tls16-exact tls-gpu --repeats 7 --max-run-seconds 7200
# E. the other tables with times
[ -e "$directory/STOP" ] && { echo "stopped before profile_split"; exit 0; }
[ -e "$directory/profile_split.json" ] || { log profile_split start; taskset -c 0-15 uv run python paper/scripts/profile_split.py --json "$directory/profile_split.json"; log profile_split end; }
[ -e "$directory/STOP" ] && { echo "stopped before screening_stats"; exit 0; }
[ -e "$directory/screening_stats.json" ] || { log screening_stats start; taskset -c 0-15 uv run python paper/scripts/screening_stats.py --json "$directory/screening_stats.json"; log screening_stats end; }
# F. the rows on one thread, last: a stopped session has all the device rows
part slow20 $bench --sizes 20000 --rows astropy tls1 tls1-exact --repeats 7
part slow $bench --sizes 100000 200000 --rows astropy tls1 tls1-exact --repeats 7 --max-run-seconds 1000
echo "session done"
