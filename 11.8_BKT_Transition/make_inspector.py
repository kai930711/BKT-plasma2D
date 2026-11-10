"""Build the interactive BKT run inspector (web/BKT_Inspector.html + web/data/rho_<i>.txt, base64).

    /opt/anaconda3/bin/python make_inspector.py

Needs Results/states_labelled.csv, Results/bkt_lines.csv (bkt_criterion.py) and the unpacked runs.
Snapshot files: per density, states ordered (T* ascending, then E0* ascending); per state
int8 charges[N], then uint16 coords[F][N][2] with x = u / 65536 (F = every 3rd stored frame).
"""
import base64
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from read_scan_v2 import list_runs, load_run

DATA = Path('/Users/kai/Downloads/11.8_data/runs')
OUT = HERE / 'web'
FRAME_STEP = 3                      # keep frames 0, 3, ..., 99  -> 34 frames
TEMPLATE = HERE / 'inspector_template.html'


def main():
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(HERE / 'Results' / 'states_labelled.csv')
    li = pd.read_csv(HERE / 'Results' / 'bkt_lines.csv')
    runs = list_runs(DATA).set_index('run_id')
    rhos = sorted(st.rho_star.unique())
    Ts = sorted(st.T_star.unique())
    Es = sorted(st.E0_star.unique())
    N = int(st.N.iloc[0])
    code = {'D': 0, 'M': 1, 'C': 2, 'drop': 3, 'drift': 4}
    payload = dict(N=N, rho=rhos, T=[round(t, 4) for t in Ts], E0=[round(e, 4) for e in Es],
                   frame_step=FRAME_STEP, densities=[])
    for i, rho in enumerate(rhos):
        g = st[np.isclose(st.rho_star, rho)].sort_values(['T_star', 'E0_star']).reset_index(drop=True)
        assert len(g) == len(Ts) * len(Es)
        blob = bytearray()
        sweep_per_frame = []
        for _, s in g.iterrows():
            r = runs.loc[s.run_id]
            d = load_run(r)
            X = d['coordinates'][::FRAME_STEP]
            q = d['charges'].astype(np.int8)
            u = (np.rint(X * 65536).astype(np.int64) % 65536).astype('<u2')
            blob += q.tobytes() + u.tobytes()
            s_every = max(1, int(r.n_samples) // 100)
            sweep_per_frame.append(int(s_every * int(r.sample_every) * FRAME_STEP))
        nF = len(X)
        # base64 text: artifacts serve .txt but not raw binary; the page decodes it back to bytes
        (OUT / 'data' / f'rho_{i}.txt').write_bytes(base64.b64encode(bytes(blob)))
        L = li[np.isclose(li.rho_star, rho)].sort_values('E0_star')
        rd = lambda c, k=4: [None if pd.isna(v) else round(float(v), k) for v in g[c]]
        payload['densities'].append(dict(
            rho=rho, a=float(np.sqrt(rho / N)), frames=nF, bytes=len(blob),
            label=[code[x] for x in g.label], inv_eps=rd('inv_eps'), sem=rd('inv_eps_sem', 5),
            h1=rd('inv_eps_h1'), h2=rd('inv_eps_h2'), h1s=rd('inv_eps_h1_sem', 5), h2s=rd('inv_eps_h2_sem', 5),
            lc=rd('lc_frac', 3), lcmax=rd('lc_frac_max', 3), acc=rd('acc', 5), neff=rd('n_eff', 0),
            snn=[round(float(v) / N, 3) for v in g.SNN_m1], sweeps=[int(v) for v in g.n_sweeps],
            sweep_per_frame=sweep_per_frame,
            lines=dict(verdict=list(L.verdict), T_BKT=[None if pd.isna(v) else round(float(v), 5) for v in L.T_BKT],
                       T_err=[None if pd.isna(v) else round(float(v), 5) for v in L.T_BKT_err],
                       T_lo=[None if pd.isna(v) else float(v) for v in L.T_lo],
                       T_hi=[None if pd.isna(v) else float(v) for v in L.T_hi])))
        print(f'rho*={rho:<8} {len(blob)/1e6:.2f} MB, {nF} frames')
    html = TEMPLATE.read_text().replace('/*__DATA__*/null', json.dumps(payload, separators=(',', ':')))
    (OUT / 'BKT_Inspector.html').write_text(html)          # artifact source (skeleton added at publish)
    local = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
             '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>\n'
             + html + '\n</body></html>\n')
    (OUT / 'index.html').write_text(local)                  # local copy: serve web/ over http and open it
    print('wrote', OUT / 'BKT_Inspector.html', 'and index.html', f'({len(html)/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
