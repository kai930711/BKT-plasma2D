# 10.27_Programs_tentative

A collection of copies of the programs used so far for the BKT / liquid–gas S_QQ work.
Nothing here was moved. The originals are still in their own folders, and the scripts
in this folder still point at those original paths (see "Running" at the end).

Run Python with `/opt/anaconda3/bin/python`. Julia comes from the project's own `Project.toml`.

## 1_921_density_scan  (from `9.21_Density Correlation`)
The N = 50 scan: 10 densities, E0* 0–0.5, T* 0.15–0.35.
- `Setup.jl`, `Plasma_Interaction.jl`, `Monte_Carlo.jl` — the simulation: hard core, Ewald log-Coulomb plus the cosh term, Metropolis moves.
- `scan_921.sub`, `run_scan_921.sh` — the HTCondor submit file and its runner.
- `read_scan.py` — reads a batch (`states.csv`, `coordinates.f64`, `charges.f64`).
- `compute_structure_factors.py` — S_NN, S_QQ and 1/ε per run, written to `Results/structure_factors.csv`.
- `one_over_epsilon0.jl`, `Task_Animation_Dielectric.jl` — earlier 1/ε and animation work.
- `Analysis.ipynb`, `S_NN_Liquid_Gas.ipynb`, `Dielectric_Phase_Diagram.ipynb` — analysis notebooks.

## 2_1021_bkt_focused  (from `10.21_BKT_Focused`)
- `BKT_Focused_N50.ipynb` — the focused BKT analysis.
- `make_bkt_page.py`, `bkt_page_template.html` — build the earlier BKT map page.

## 3_1027_wide_scan  (from `10.27_Wide_SQQ_Scan`)
The wide pilot (ρ* = 0.05, T* 0.05–0.55, E0* 0–1.0, hot and cold starts) and the transition-lines page.
- `Monte_Carlo.jl` — modified copy (adds the start option and step adaptation). `Monte_Carlo.orig.jl` is the unmodified version.
- `Setup.jl`, `Plasma_Interaction.jl` — identical to the 9.21 files (checked with `cmp`).
- `Wide_Scan.jl`, `test_monte_carlo.jl` — the scan driver and its test.
- `wide_scan.sub`, `run_wide_scan.sh` — the HTCondor submit file (24 jobs, not yet submitted) and its runner.
- `compute_sqq_wide.py` — S_QQ(k), S_NN(k) and 1/ε(k) for the pilot.
- `read_scan.py` — the same reader as in 9.21.
- `make_transition_page.py`, `transition_page_template.html` — build `Transition_Lines.html`, the published page.

## 4_analysis_scripts  (from the session scratchpad)
Scripts that wrote the figures in `10.27_Wide_SQQ_Scan/Results_pilot`.
- `phase_map.py`, `phase_map2.py` — S_QQ phase maps, T* on y and E0* on x.
- `stack.py`, `stack_all.py` — the BKT line stacked on the S_QQ step estimate, for ρ* = 0.05 and for all densities.
- `alongE0.py` — steepest change of 1/ε along E0* at fixed T*.
- `refcheck.py` — pilot S_QQ(k), 1/ε(k) and S_NN(k) against the reference figure.
- `feat.py`, `classify.py`, `classmap.py` — the S_QQ(k)-shape features, the three-class classifier and its map.
- `lines.py` — the BKT and liquid–gas transition-line plots.
- `jump_test.py`, `k_coverage_fig.py`, `sqq_only_test.py`, `sqq_only_test2.py` — earlier checks: step tests on the 9.21 table, the k-coverage figure, and tests of whether S_QQ alone predicts clustering.

## 5_reports
- `make_status_pdf.py` — builds `Status_and_Limitations.pdf` (needs `reportlab`; I used a throwaway venv).
- `Status_and_Limitations.pdf` — the 2-page progress and limitations summary (dated 28 October 2026).

## Not included
- Simulation data and result files (Data/, Results/, Results_pilot/, the downloaded `data_n100_200` batches).
- The new N = 100 and N = 200 data has no program yet. It is unprocessed.

## Running
Copies in this folder have not been tested from here. Several scripts use absolute paths into
`9.21_Density Correlation`, `10.27_Wide_SQQ_Scan` and the session scratchpad. Run them from the
original folders, or edit the paths first.
