#!/usr/bin/env bash
set -euo pipefail

J="$1"
mkdir -p results
RESULT_ARCHIVE=Dielectric_results.tar.gz
CHECKPOINT_SECONDS=900
CHECKPOINT_PID=""

printf 'Job=%s\nRuns (job,state_id,replicate_id,N,rho_star,T_star,E0_star,...):\n' "$J"
awk -F, -v j="$J" 'NR>1 && $1==j' Dielectric_jobs.csv
printf 'Host=%s\nWorking_directory=%s\n' "$(hostname)" "$PWD"
echo "Result archive checkpoint interval: ${CHECKPOINT_SECONDS}s."
echo "Live progress: follow this job's streamed .out file and watch START/COMPLETE lines."

snapshot_results() {
    local reason="${1:-checkpoint}"
    local tmp="${RESULT_ARCHIVE}.tmp.$$"
    if tar -czf "$tmp" -C results .; then
        mv -f "$tmp" "$RESULT_ARCHIVE"
        printf '[%s] Result snapshot ready (%s): %s\n' "$(date -u +%FT%TZ)" "$reason" "$RESULT_ARCHIVE"
        return 0
    fi
    rm -f "$tmp"
    printf '[%s] WARNING: result snapshot failed (%s); keeping previous archive.\n' "$(date -u +%FT%TZ)" "$reason" >&2
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
snapshot_results "initial empty archive"

JULIA_ARCHIVE=julia-1.12.6-linux-x86_64.tar.gz
printf '[%s] Extracting Julia...\n' "$(date -u +%FT%TZ)"
tar -xzf "$JULIA_ARCHIVE"
JULIA="$PWD/julia-1.12.6/bin/julia"
export JULIA_DEPOT_PATH="$PWD/julia_depot"
export JULIA_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$JULIA_DEPOT_PATH"

printf '[%s] Installing Julia dependencies...\n' "$(date -u +%FT%TZ)"
"$JULIA" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'

printf '[%s] Beginning simulation; follow the START and COMPLETE lines below.\n' "$(date -u +%FT%TZ)"
checkpoint_loop &
CHECKPOINT_PID=$!
"$JULIA" --startup-file=no --project=. Output_Coordinate_Dielectric.jl --job "$J" \
    --jobs-file Dielectric_jobs.csv --output-root "$PWD/results" | tee results/simulation_progress.log
