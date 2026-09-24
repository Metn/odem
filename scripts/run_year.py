#!/usr/bin/env python
"""Run one ODEM scheme-year as parallel single-month runs and merge them.

Each month is an independent `python odem.py` process (the model is a pure
per-timestep computation, so months do not interact), N run at once, then
scripts/merge_year.py assembles the run directory. Twelve MERRA-2-grid months
take about 25 minutes in parallel on a desktop; ERA5-grid months 2 to 5 hours.

Usage (all odem.py options after `--` are passed through):
  python scripts/run_year.py --forcing merra2 --meteo-dir E:/.../merra2_monthly_v2 \
      --year 2006 --out E:/.../output_v2/k14_m2_2006 --workers 12 -- \
      --scheme k14 --soilgrids ... --prigent ... --modis-lai ... --landcover ... --save-diags
Options before `--`: --months 1-12 (or 1,2,7), --workers N, --dry-run.
A month whose directory already holds a finished monthly file is skipped, so a
killed batch can be resumed by re-running the same command.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ODEM = os.path.join(os.path.dirname(HERE), 'odem.py')
MERGE = os.path.join(HERE, 'merge_year.py')


def parse_months(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--forcing', choices=['era5', 'merra2'], required=True)
    p.add_argument('--meteo-dir', required=True, help='directory with <prefix>_YYYYMM.nc monthly files')
    p.add_argument('--prefix', default=None, help='file prefix (default era5_dust or merra2_dust)')
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--months', default='1-12')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--out', required=True, help='run directory (months go to <out>/months/mMM, merged into <out>)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('odem_args', nargs=argparse.REMAINDER, help='odem.py options after --')
    a = p.parse_args()
    extra = [x for x in a.odem_args if x != '--']
    prefix = a.prefix or f'{a.forcing}_dust'
    months = parse_months(a.months)
    mroot = os.path.join(a.out, 'months')
    os.makedirs(mroot, exist_ok=True)

    jobs = []
    for m in months:
        f = os.path.join(a.meteo_dir, f'{prefix}_{a.year}{m:02d}.nc')
        if not os.path.exists(f):
            raise SystemExit(f'missing forcing file {f}')
        mdir = os.path.join(mroot, f'm{m:02d}')
        os.makedirs(mdir, exist_ok=True)
        done = glob.glob(os.path.join(mdir, f'odem_{a.year}{m:02d}.nc'))
        if done and not a.dry_run:
            print(f'  m{m:02d}: exists, skipping')
            continue
        cmd = [sys.executable, ODEM, f'--{a.forcing}', f, '--output-dir', mdir] + extra
        jobs.append((m, mdir, cmd))
    if a.dry_run:
        for m, mdir, cmd in jobs:
            print(' '.join(cmd))
        return

    def run(job):
        m, mdir, cmd = job
        t0 = time.time()
        with open(os.path.join(mdir, 'run.log'), 'w', encoding='utf-8') as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        return m, r.returncode, time.time() - t0

    print(f'{len(jobs)} months, {a.workers} workers, scheme args: {" ".join(extra)}')
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fut in as_completed(futs):
            m, rc, dt_ = fut.result()
            print(f'  m{m:02d}: exit {rc} in {dt_/60:.1f} min', flush=True)
            if rc != 0:
                failed.append(m)
    if failed:
        raise SystemExit(f'failed months: {failed}; see months/mMM/run.log')
    r = subprocess.run([sys.executable, MERGE, '--months-root', mroot, '--out', a.out])
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
