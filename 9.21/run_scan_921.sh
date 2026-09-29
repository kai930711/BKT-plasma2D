#!/usr/bin/env bash
set -euo pipefail

# HTCondor transfers the eight project files and Julia archive into its scratch
# directory. One invocation handles one density (121 consecutive global IDs).
# The MC and task files are used as-is. Progress is the task's START/COMPLETE lines.
# A periodically refreshed archive gives ON_EXIT_OR_EVICT something useful to
# return even when the execute slot is reclaimed between normal job exits.

mkdir -p results
RESULT_ARCHIVE=scan_results.tar.gz
CHECKPOINT_SECONDS="${CHECKPOINT_SECONDS:-900}"
CHECKPOINT_PID=""

snapshot_results() {
    local reason="${1:-checkpoint}"
    local tmp="${RESULT_ARCHIVE}.tmp.$$"
    if tar -czf "$tmp" -C results .; then
        mv -f "$tmp" "$RESULT_ARCHIVE"
        printf '[%s] Result snapshot ready (%s): %s\n' \
            "$(date -u +%FT%TZ)" "$reason" "$RESULT_ARCHIVE"
        return 0
    fi
    rm -f "$tmp"
    printf '[%s] WARNING: result snapshot failed (%s); keeping previous archive.\n' \
        "$(date -u +%FT%TZ)" "$reason" >&2
    return 1
}

checkpoint_loop() {
    while sleep "$CHECKPOINT_SECONDS"; do
        snapshot_results "periodic ${CHECKPOINT_SECONDS}s checkpoint" || true
    done
}

finish() {
    local status=$?
    trap - EXIT
    if [[ -n "$CHECKPOINT_PID" ]]; then
        kill "$CHECKPOINT_PID" 2>/dev/null || true
        wait "$CHECKPOINT_PID" 2>/dev/null || true
    fi
    printf '[%s] Packing results; simulation exit status=%s\n' "$(date -u +%FT%TZ)" "$status"
    # The Julia writer publishes progress.csv only after a point's binary/CSV data
    # are flushed. A snapshot may contain an uncommitted trailing point, but the
    # supplied read_scan.py ignores data beyond the last committed progress row.
    if ! snapshot_results "final"; then
        echo 'ERROR: failed to refresh the final result archive.' >&2
        if (( status == 0 )); then status=74; fi
    fi
    printf '[%s] Job finished; exit_status=%s\n' "$(date -u +%FT%TZ)" "$status"
    exit "$status"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# Ensure the declared transfer file exists even if setup later fails.
snapshot_results "initial empty archive"

if (( $# != 3 )); then
    echo 'Usage: run_scan_921.sh RHO_TAG START_ID END_ID' >&2
    exit 2
fi
RHO_TAG="$1"
START_ID="$2"
END_ID="$3"
case "$RHO_TAG:$START_ID:$END_ID" in
    0p001:1:121)     RHO_STAR=0.001 ;;
    0p002:122:242)   RHO_STAR=0.002 ;;
    0p005:243:363)   RHO_STAR=0.005 ;;
    0p01:364:484)    RHO_STAR=0.01 ;;
    0p02:485:605)    RHO_STAR=0.02 ;;
    0p03:606:726)    RHO_STAR=0.03 ;;
    0p05:727:847)    RHO_STAR=0.05 ;;
    0p07:848:968)    RHO_STAR=0.07 ;;
    0p1:969:1089)    RHO_STAR=0.1 ;;
    0p2:1090:1210)   RHO_STAR=0.2 ;;
    *) echo 'ERROR: density tag and state range do not match the 1210-point grid.' >&2; exit 2 ;;
esac

printf '[%s] Starting density rho*=%s, states %s..%s (121 points)\n' \
    "$(date -u +%FT%TZ)" "$RHO_STAR" "$START_ID" "$END_ID"
printf 'Host=%s\nWorking_directory=%s\n' "$(hostname)" "$PWD"
echo 'Settings: N=50, burn-in=5000, production=10000, sample_every=10, replicate_id=1.'
echo 'The task prints START/COMPLETE for each point; completed points are saved immediately.'
echo "Result archive checkpoint interval: ${CHECKPOINT_SECONDS}s."
echo "Live progress: follow this job's streamed .out file and watch START/COMPLETE lines."

JULIA_ARCHIVE=julia-1.12.6-linux-x86_64.tar.gz
for file in "$JULIA_ARCHIVE" Project.toml Manifest.toml Setup.jl Plasma_Interaction.jl \
    Monte_Carlo.jl one_over_epsilon0.jl Task_Animation_Dielectric.jl read_scan.py; do
    if [[ ! -r "$file" ]]; then
        printf 'ERROR: required input is missing or unreadable: %s\n' "$file" >&2
        exit 2
    fi
done

printf '[%s] Extracting Julia...\n' "$(date -u +%FT%TZ)"
tar -xzf "$JULIA_ARCHIVE"
JULIA="$PWD/julia-1.12.6/bin/julia"
if [[ ! -x "$JULIA" ]]; then
    echo 'ERROR: archive did not provide julia-1.12.6/bin/julia.' >&2
    exit 2
fi
export JULIA_DEPOT_PATH="$PWD/julia_depot"
export JULIA_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$JULIA_DEPOT_PATH"
"$JULIA" --version

# Same dependency installation approach as the earlier cluster submission.
# First-time installation requires package-server access from the execution node.
printf '[%s] Installing Julia dependencies...\n' "$(date -u +%FT%TZ)"
"$JULIA" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'

TASK_ARGS=(--start-id "$START_ID" --end-id "$END_ID" --replicate-id 1
    --burnin 5000 --sweeps 10000 --sample-every 10 --output-root "$PWD/results")
"$JULIA" --startup-file=no --project=. Task_Animation_Dielectric.jl "${TASK_ARGS[@]}" --dry-run
printf '[%s] Beginning simulation; follow the START and COMPLETE lines below.\n' "$(date -u +%FT%TZ)"
# Keep a copy of stdout inside the returned archive while still streaming it to
# HTCondor's .out file in real time. pipefail preserves Julia's nonzero status.
checkpoint_loop &
CHECKPOINT_PID=$!
"$JULIA" --startup-file=no --project=. Task_Animation_Dielectric.jl "${TASK_ARGS[@]}" \
    | tee results/simulation_progress.log
