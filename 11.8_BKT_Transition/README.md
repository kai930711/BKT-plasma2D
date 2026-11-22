# 11.8_BKT_Transition

BKT analysis of the 11.8 scan (N = 50, 11 densities ρ* = 0.001–0.1, T* = 0.05–0.30 step 0.01,
E0* = 0–0.5 step 0.02; 7,436 runs). Only the BKT transition is analysed here.

Raw data: `/Users/kai/Downloads/11.8.tar`, unpacked to `/Users/kai/Downloads/11.8_data/runs` (3.1 GB).

## Criterion (in one paragraph)
d = 1/ε(k_min) − 4T*, with 1/ε = 1 − κ S_QQ(m²=1)/(2π), S_QQ without 1/N, L = 1. A state counts only if no
snapshot has a cluster (r < 1.5σ) holding ≥ N/2 particles (no drift check: it never moved T_BKT). Valid states are
labelled D (d > 2σ), C (d < −2σ) or M. A (ρ*, E0*) line has a BKT transition if its D/C labels read D…D C…C going up
in T* with no excluded state inside the bracket; T_BKT is the zero of a weighted line fit of d(T*) across the bracket.
Full statement: the docstring of `bkt_criterion.py` and section "The criterion" of the notebook.

## Files
- `read_scan_v2.py` — reader for the schema-v2 output (`series.f64`, `coordinates.f64`, `charges.f64`).
- `compute_dielectric.py` — per run: 1/ε for 10 k-shells, block errors, half-run drift, largest cluster per snapshot.
  Writes `Results/runs_11p8.csv` (≈1.5 min with all cores).
- `bkt_criterion.py` — the criterion. Writes `Results/states_labelled.csv` and `Results/bkt_lines.csv`.
- `bkt_plots.py` — `fig1_cuts.png`, `fig2_state_maps.png`, `fig3_bkt_lines.png` in `Results/`.
- `make_notebook.py` — builds `BKT_Transition_11p8.ipynb` (criterion, data checks, figures, summary table,
  threshold sensitivity). `Results/bkt_summary_by_density.csv` is written by the notebook.

## Run inspector (web view)
Published page: https://claude.ai/artifact/B5ehEgUJU6xuZNYwFutUK3 (private to you until shared).
- `make_inspector.py` + `inspector_template.html` build `web/BKT_Inspector.html` (artifact source), `web/index.html`
  (local copy) and `web/data/rho_<i>.txt` (34 snapshots per run, base64 uint16; artifacts don't serve raw binary).
- Local use: `/opt/anaconda3/bin/python -m http.server 8811 --directory web`, then open http://localhost:8811
  (opening the file directly cannot load the snapshots).
- Republishing needs two publishes (65 MB of snapshot files > 64 MB per publish).
- `web/data/` (65 MB, generated) is in git so the page works from a clone; `make_inspector.py` rebuilds it from the raw data.

## Running
```
/opt/anaconda3/bin/python compute_dielectric.py          # only if runs_11p8.csv is missing
/opt/anaconda3/bin/python bkt_criterion.py
/opt/anaconda3/bin/python bkt_plots.py
/opt/anaconda3/bin/python make_notebook.py
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace BKT_Transition_11p8.ipynb
```
