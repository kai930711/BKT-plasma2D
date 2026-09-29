"""Read a 9.21 batch without Julia. Requires NumPy and Python 3.11+.

    from read_scan import load_run
    data = load_run('/path/to/scan_921_XXXX', state_id=1)
    R = data['coordinates']  # shape (frames, N, 2), Float64, read-only
    q = data['charges']      # shape (N,), aligned with R[:, particle, :]

Use allow_incomplete=True to read only flushed/committed frames of an interrupted
run. Never infer its valid frame count from binary file length alone. This reader
also works on a completed prefix of an active batch; it is not a resume mechanism.
"""
from pathlib import Path
import csv
import tomllib
import numpy as np


def _complete_rows(path, byte_limit=None, start_offset=None):
    with Path(path).open('rb') as f:
        header = f.readline()
        if not header.endswith(b'\n'):
            raise ValueError(f'Missing complete CSV header in {path}')
        fields = next(csv.reader([header.decode('utf-8')], strict=True))
        if start_offset is not None:
            if start_offset < f.tell() or (byte_limit is not None and start_offset > byte_limit):
                raise ValueError('Invalid CSV byte range')
            f.seek(start_offset)
        # Writer's CSV fields contain no embedded newlines. Stream only this run's
        # byte range, ignoring any truncated final row and uncommitted trailing data.
        while byte_limit is None or f.tell() < byte_limit:
            line = f.readline()
            if not line or not line.endswith(b'\n'):
                break
            if byte_limit is not None and f.tell() > byte_limit:
                raise ValueError('Commit offset is not at a CSV row boundary')
            values = next(csv.reader([line.decode('utf-8')], strict=True))
            if len(values) != len(fields):
                raise ValueError(f'Malformed complete CSV row in {path}')
            yield dict(zip(fields, values))


def load_run(batch_dir, state_id, replicate_id=1, *, allow_incomplete=False):
    root = Path(batch_dir)
    with (root / 'metadata.toml').open('rb') as f:
        metadata = tomllib.load(f)
    if metadata['schema_version'] != 1 or metadata['coordinate_dtype'] != 'Float64':
        raise ValueError('Unsupported data schema')
    dtype = np.dtype({'little': '<f8', 'big': '>f8'}[metadata['byte_order']])
    matches = [r for r in _complete_rows(root / 'states.csv')
               if int(r['state_id']) == state_id and int(r['replicate_id']) == replicate_id]
    if len(matches) != 1:
        raise ValueError('Expected exactly one matching run in this batch')
    params = matches[0]
    commits = [r for r in _complete_rows(root / 'progress.csv') if r['run_id'] == params['run_id']]
    if not commits:
        raise ValueError('Run has no committed data yet')
    progress = commits[-1]
    if progress['status'] != 'complete' and not allow_incomplete:
        raise ValueError('Run is incomplete; use allow_incomplete=True to inspect its committed prefix')
    n = int(params['N'])
    m = int(progress['n_samples_committed'])
    planned = int(params['n_samples_planned'])
    if not 0 <= m <= planned or (progress['status'] == 'complete' and m != planned):
        raise ValueError('Invalid committed frame count')
    coord_start = int(params['coordinates_offset_bytes'])
    charge_start = int(params['charges_offset_bytes'])
    coord_end = coord_start + 16*n*m
    charge_end = charge_start + 8*n
    if coord_end != int(progress['coordinates_end_bytes']) or charge_end != int(progress['charges_end_bytes']):
        raise ValueError('Binary offsets do not match committed counts')
    for name, end in [('coordinates.f64', coord_end), ('charges.f64', charge_end),
                      ('samples.csv', int(progress['samples_end_bytes']))]:
        if (root / name).stat().st_size < end:
            raise ValueError(f'Truncated committed file: {name}')
    if m:
        # Disk: frame, axis, particle; exposed: frame, particle, axis.
        coords = np.memmap(root / 'coordinates.f64', dtype=dtype, mode='r',
                           offset=coord_start, shape=(m, 2, n)).transpose(0, 2, 1)
    else:
        coords = np.empty((0, n, 2), dtype=dtype)
    charges = np.fromfile(root / 'charges.f64', dtype=dtype, count=n, offset=charge_start)
    if not np.isin(charges, [-1, 1]).all() or charges.sum() != 0:
        raise ValueError('Invalid charge array')
    rows = list(_complete_rows(root / 'samples.csv', int(progress['samples_end_bytes']),
                               int(progress['samples_start_bytes'])))
    if any(r['run_id'] != params['run_id'] or int(r['state_id']) != state_id for r in rows):
        raise ValueError('Sample byte range contains data from another run')
    if len(rows) != m:
        raise ValueError('Energy/coordinate frame count mismatch')
    sample = np.array([int(r['sample']) for r in rows], dtype=np.int64)
    production = np.array([int(r['production_sweep']) for r in rows], dtype=np.int64)
    total = np.array([int(r['total_sweep']) for r in rows], dtype=np.int64)
    if (not np.array_equal(sample, np.arange(1, m+1))
            or not np.array_equal(production, 1 + np.arange(m)*int(params['sample_every']))
            or not np.array_equal(total, production + int(params['n_burnin']))):
        raise ValueError('Unexpected frame ordering or sweep numbers')
    energy = np.array([[float(r[k]) for k in ('U1', 'U2', 'beta_U')] for r in rows]).reshape(m, 3)
    if not np.isfinite(energy).all() or not np.allclose(
            energy[:, 2], float(params['kappa'])*energy[:, 0]+energy[:, 1], rtol=1e-10, atol=1e-9):
        raise ValueError('Invalid energy sequence')
    return dict(parameters=params, metadata=metadata, progress=progress, coordinates=coords,
                charges=charges, sample=sample, production_sweep=production, total_sweep=total,
                U1=energy[:, 0], U2=energy[:, 1], beta_U=energy[:, 2])
