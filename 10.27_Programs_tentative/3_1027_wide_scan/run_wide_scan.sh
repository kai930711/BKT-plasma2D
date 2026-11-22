#!/usr/bin/env bash
set -euo pipefail

# Wide (T*, E0*) scan at fixed N and rho*. HTCondor transfers the project files and the Julia
# archive into its scratch directory. One invocation = one replicate (seed + start type) over one
# block of consecutive state IDs of the 483-point grid. Progress is the task's START/COMPLETE lines.
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

if (( $# != 5 )); then
    echo 'Usage: run_wide_scan.sh REPLICATE_ID INIT(hot|cold) START_ID END_ID TAG' >&2
    exit 2
fi
REPLICATE_ID="$1"
INIT="$2"
START_ID="$3"
END_ID="$4"
TAG="$5"
case "$INIT" in hot|cold) ;; *) echo 'INIT must be hot or cold' >&2; exit 2 ;; esac
N_PART=50
RHO_STAR=0.05
T_RANGE=0.05:0.025:0.60
E0_RANGE=0:0.05:1.0

printf '[%s] Starting %s: replicate=%s init=%s states %s..%s of the 483-point grid\n' \
    "$(date -u +%FT%TZ)" "$TAG" "$REPLICATE_ID" "$INIT" "$START_ID" "$END_ID"
printf 'Host=%s\nWorking_directory=%s\n' "$(hostname)" "$PWD"
echo "Settings: N=$N_PART, rho*=$RHO_STAR, T*=$T_RANGE, E0*=$E0_RANGE, burn-in=5000, production=10000, sample_every=10, adaptive step (burn-in only)."

JULIA_ARCHIVE=julia-1.12.6-linux-x86_64.tar.gz
for file in "$JULIA_ARCHIVE" Project.toml Manifest.toml Setup.jl Plasma_Interaction.jl \
    Monte_Carlo.jl Wide_Scan.jl read_scan.py; do
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

TASK_ARGS=(--N "$N_PART" --rho "$RHO_STAR" --T-range "$T_RANGE" --E0-range "$E0_RANGE"
    --init "$INIT" --replicate-id "$REPLICATE_ID" --start-id "$START_ID" --end-id "$END_ID"
    --burnin 5000 --sweeps 10000 --sample-every 10 --output-root "$PWD/results")
"$JULIA" --startup-file=no --project=. Wide_Scan.jl "${TASK_ARGS[@]}" --dry-run
printf '[%s] Beginning simulation; follow the START and COMPLETE lines below.\n' "$(date -u +%FT%TZ)"
# Keep a copy of stdout inside the returned archive while still streaming it to
# HTCondor's .out file in real time. pipefail preserves Julia's nonzero status.
checkpoint_loop &
CHECKPOINT_PID=$!
"$JULIA" --startup-file=no --project=. Wide_Scan.jl "${TASK_ARGS[@]}" \
    | tee results/simulation_progress.log
