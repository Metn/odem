#!/usr/bin/env python
"""Merge per-month ODEM output directories into one run directory with the
files a single multi-month run would have written: odem_YYYYMM.nc for each
month (moved or copied), odem_annual_mean.nc (time mean and max of F_d over
all timesteps, weighted by each month's timestep count) and, when every
month has one, odem_diag_mean.nc (timestep-weighted mean of each diagnostic,
static fields copied from the first month).

Usage:
  python scripts/merge_year.py --months-root <dir with m01 .. m12> --out <run dir> [--copy]

The month directories are what scripts/run_year.py writes. The merged annual
mean equals the one run_model() writes for a full-year run up to the order
of floating-point accumulation (relative 1e-15), so the regression harness
budget check applies unchanged.
"""
import argparse
import datetime as dt
import glob
import os
import shutil

import numpy as np
import netCDF4 as nc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--months-root', required=True, help='directory holding m01, m02, ... month run dirs')
    p.add_argument('--out', required=True, help='merged run directory (created)')
    p.add_argument('--copy', action='store_true', help='copy monthly files instead of moving them')
    a = p.parse_args()
    root, out = a.months_root, a.out
    os.makedirs(out, exist_ok=True)
    mdirs = sorted(d for d in glob.glob(os.path.join(root, 'm[0-9][0-9]')) if os.path.isdir(d))
    if not mdirs:
        raise SystemExit(f'no month directories under {root}')

    F_sum = None
    F_max = None
    n_total = 0
    t_start = t_end = None
    diag_sum = {}
    diag_ok = True
    static_src = None
    monthly_files = []
    for d in mdirs:
        mfiles = sorted(glob.glob(os.path.join(d, 'odem_[0-9]*.nc')))
        if len(mfiles) != 1:
            raise SystemExit(f'{d}: expected one monthly file, found {mfiles}')
        mf = mfiles[0]
        with nc.Dataset(mf) as ds:
            F = np.nan_to_num(np.asarray(ds.variables['F_d'][:], dtype=np.float64))
            lon = np.array(ds.variables['lon'][:]); lat = np.array(ds.variables['lat'][:])
            tv = ds.variables['time']
            times = nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
        n = F.shape[0]
        if F_sum is None:
            F_sum = np.zeros(F.shape[1:], dtype=np.float64)
            F_max = np.zeros(F.shape[1:], dtype=np.float32)
        F_sum += F.sum(axis=0)
        F_max = np.maximum(F_max, F.max(axis=0).astype(np.float32))
        n_total += n
        t_start = times[0] if t_start is None else min(t_start, times[0])
        t_end = times[-1] if t_end is None else max(t_end, times[-1])
        monthly_files.append(mf)
        # diagnostics: merged only when every month has a diag file
        df = os.path.join(d, 'odem_diag_mean.nc')
        if not os.path.exists(df):
            diag_ok = False
        elif diag_ok:
            with nc.Dataset(df) as ds:
                nt = int(ds.n_timesteps)
                for name, v in ds.variables.items():
                    if name in ('lat', 'lon'):
                        continue
                    if getattr(v, 'missing_value', None) is not None:
                        continue  # static field, copied from the first month
                    arr = np.asarray(v[:], dtype=np.float64) * nt
                    diag_sum[name] = diag_sum.get(name, 0.0) + arr
            static_src = static_src or df
        print(f'  {os.path.basename(d)}: {n} steps, month sum {F.sum():.6e}')

    moved = []
    for mf in monthly_files:
        dst = os.path.join(out, os.path.basename(mf))
        if os.path.abspath(mf) != os.path.abspath(dst):
            (shutil.copy2 if a.copy else shutil.move)(mf, dst)
        moved.append(dst)
    monthly_files = moved

    F_mean = F_sum / n_total
    fn = os.path.join(out, 'odem_annual_mean.nc')
    with nc.Dataset(fn, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', len(lat)); ds.createDimension('lon', len(lon))
        v = ds.createVariable('lat', 'f8', ('lat',)); v[:] = lat; v.units = 'degrees_north'
        v = ds.createVariable('lon', 'f8', ('lon',)); v[:] = lon; v.units = 'degrees_east'
        v = ds.createVariable('F_d_mean', 'f4', ('lat', 'lon'), zlib=True, complevel=4)
        v[:] = F_mean.astype(np.float32); v.long_name = 'Time-mean dust emission flux'; v.units = 'kg m-2 s-1'
        v = ds.createVariable('F_d_max', 'f4', ('lat', 'lon'), zlib=True, complevel=4)
        v[:] = F_max; v.long_name = 'Maximum dust emission flux'; v.units = 'kg m-2 s-1'
        with nc.Dataset(monthly_files[0]) as m0:
            ds.title = 'ODEM — Offline Dust Emission Model — annual mean'
            ds.source = getattr(m0, 'source', '')
            ds.scheme = getattr(m0, 'scheme', '')
        ds.period = f'{t_start} to {t_end}'
        ds.n_timesteps = str(n_total)
        ds.merged_from = f'{len(monthly_files)} monthly runs by scripts/merge_year.py'
        ds.history = f'Created {dt.datetime.now().isoformat()}'
    # global total, same cell-area formula as odem_regression.py
    dlat = abs(lat[1] - lat[0]); dlon = abs(lon[1] - lon[0])
    cell = (dlat * np.pi / 180) * (dlon * np.pi / 180) * 6.371e6 ** 2 * np.cos(np.deg2rad(lat))
    area = cell[:, None] * np.ones((1, len(lon)))
    total = float((F_mean * area).sum() * 365.25 * 24 * 3600 / 1e9)
    print(f'wrote {fn}: {n_total} timesteps, annual total {total:.1f} Tg/yr')

    if diag_ok and diag_sum and static_src:
        fn = os.path.join(out, 'odem_diag_mean.nc')
        with nc.Dataset(static_src) as src, nc.Dataset(fn, 'w', format='NETCDF4') as ds:
            ds.createDimension('lat', len(lat)); ds.createDimension('lon', len(lon))
            v = ds.createVariable('lat', 'f8', ('lat',)); v[:] = lat; v.units = 'degrees_north'
            v = ds.createVariable('lon', 'f8', ('lon',)); v[:] = lon; v.units = 'degrees_east'
            for name, arr in diag_sum.items():
                sv = src.variables[name]
                v = ds.createVariable(name, 'f4', ('lat', 'lon'), zlib=True, complevel=4)
                v[:] = (arr / n_total).astype(np.float32)
                for k in sv.ncattrs():
                    v.setncattr(k, sv.getncattr(k))
            for name, sv in src.variables.items():
                if name in ('lat', 'lon') or name in diag_sum:
                    continue
                v = ds.createVariable(name, sv.dtype, sv.dimensions, zlib=True, complevel=4)
                v[:] = sv[:]
                for k in sv.ncattrs():
                    v.setncattr(k, sv.getncattr(k))
            for k in src.ncattrs():
                ds.setncattr(k, src.getncattr(k))
            ds.period = f'{t_start} to {t_end}'
            ds.n_timesteps = n_total
            ds.merged_from = f'{len(monthly_files)} monthly runs by scripts/merge_year.py'
        print(f'wrote {fn}')


if __name__ == '__main__':
    main()
