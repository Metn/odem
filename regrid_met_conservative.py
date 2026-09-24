#!/usr/bin/env python3
"""
regrid_met_conservative.py -- put an ERA5 monthly meteorology file onto the
MERRA-2 grid with the conservative (area-weighted) regrid from odem.py.

Every data variable of the source file (all timesteps unless --steps N) is
regridded onto the latitude/longitude grid of the target-grid file. The output
keeps the SAME variable and dimension names as the ERA5 input
(valid_time / latitude / longitude, zust, swvl1, ...) so odem.py reads it
unchanged through --era5. Variable attributes are copied; global attributes
record the method, the source file, the target grid and the date.

Usage:
    python regrid_met_conservative.py --src <ERA5 monthly file> \
        --target-grid <MERRA-2 file> --out <path> [--steps N] [--chunk 8]

Notes:
  - lsm (if present) becomes a land FRACTION (area mean of the source mask),
    which odem.py thresholds at 0.5 as before; noted in the variable attributes.
  - slt (ERA5 soil type, categorical) is area-averaged like everything else;
    odem.py does not use it. Noted in the variable attributes.
  - Time is processed in chunks of --chunk steps to keep memory modest.
  - Output is float32 like the input, zlib level 1, one timestep per chunk.

Author: Metin Baykara
"""

import sys
import io
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import netCDF4 as nc4

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from odem import ConservativeRegridder  # noqa: E402


def _axis(ds, names):
    for n in names:
        if n in ds.variables:
            return n, np.array(ds.variables[n][:], dtype=np.float64)
    raise KeyError(f"None of {names} found in {ds.filepath()}")


def regrid_file(src_path, target_grid_path, out_path, steps=None, chunk=8,
                complevel=1):
    src_path = Path(src_path)
    target_grid_path = Path(target_grid_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc4.Dataset(str(target_grid_path), 'r') as tg:
        _, tgt_lat = _axis(tg, ['lat', 'latitude'])
        _, tgt_lon = _axis(tg, ['lon', 'longitude'])

    src = nc4.Dataset(str(src_path), 'r')
    lat_name, src_lat = _axis(src, ['latitude', 'lat'])
    lon_name, src_lon = _axis(src, ['longitude', 'lon'])
    time_name = 'valid_time' if 'valid_time' in src.dimensions else 'time'
    nt_src = len(src.dimensions[time_name])
    nt = nt_src if steps is None else min(int(steps), nt_src)

    print(f"Source:      {src_path}")
    print(f"Target grid: {target_grid_path}")
    print(f"Output:      {out_path}")
    print(f"Grid: {src_lat.size} x {src_lon.size} -> {tgt_lat.size} x {tgt_lon.size}; "
          f"timesteps {nt}/{nt_src}; chunk {chunk}")

    t0 = time.time()
    rg = ConservativeRegridder(src_lon, src_lat, tgt_lon, tgt_lat)
    print(f"Weights built in {time.time() - t0:.2f} s "
          f"(nnz lat {rg.A_lat.nnz}, lon {rg.A_lon.nnz})")

    out = nc4.Dataset(str(out_path), 'w', format='NETCDF4')
    try:
        # Dimensions: same names as the source; time limited to nt.
        for dname, dim in src.dimensions.items():
            if dname == lat_name:
                out.createDimension(dname, tgt_lat.size)
            elif dname == lon_name:
                out.createDimension(dname, tgt_lon.size)
            elif dname == time_name:
                out.createDimension(dname, nt)
            else:
                out.createDimension(dname, None if dim.isunlimited() else len(dim))

        # Global attributes: copy, then record what was done.
        for a in src.ncattrs():
            out.setncattr(a, src.getncattr(a))
        out.regrid_method = ('conservative area-weighted (first-order): 1-D overlap '
                             'weights in sin(latitude) and longitude (periodic), '
                             'tensor product, normalised over valid source cells; '
                             'odem.ConservativeRegridder')
        out.regrid_source_file = str(src_path)
        out.regrid_source_grid = (f'{src_lat.size} x {src_lon.size}, lat {src_lat[0]} to '
                                  f'{src_lat[-1]}, lon {src_lon[0]} to {src_lon[-1]}')
        out.regrid_target_grid_file = str(target_grid_path)
        out.regrid_target_grid = (f'{tgt_lat.size} x {tgt_lon.size}, lat {tgt_lat[0]} to '
                                  f'{tgt_lat[-1]}, lon {tgt_lon[0]} to {tgt_lon[-1]}')
        out.regrid_timesteps = f'{nt} of {nt_src}'
        out.regrid_date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        out.regrid_script = 'regrid_met_conservative.py'
        hist = getattr(src, 'history', '')
        out.history = (f"{out.regrid_date} regrid_met_conservative.py conservative regrid "
                       f"onto {target_grid_path.name} grid; " + hist)

        # Coordinate and auxiliary variables
        for vname, var in src.variables.items():
            dims = var.dimensions
            if vname == lat_name:
                v = out.createVariable(vname, var.dtype, dims)
                _copy_attrs(var, v, skip=('stored_direction',))
                v[:] = tgt_lat
                if tgt_lat[0] > tgt_lat[-1]:
                    v.stored_direction = 'decreasing'
                continue
            if vname == lon_name:
                v = out.createVariable(vname, var.dtype, dims)
                _copy_attrs(var, v)
                v[:] = tgt_lon
                continue
            if lat_name in dims or lon_name in dims:
                continue   # data variables handled below
            # Anything else (time, number, expver, ...): copy, time-limited
            fill = getattr(var, '_FillValue', None)
            v = out.createVariable(vname, var.dtype, dims, fill_value=fill)
            _copy_attrs(var, v)
            if time_name in dims and var.ndim >= 1:
                v[:] = var[:nt]
            else:
                v[:] = var[:]

        # Data variables on the lat/lon grid
        data_vars = [n for n, var in src.variables.items()
                     if lat_name in var.dimensions and lon_name in var.dimensions]
        for vname in data_vars:
            var = src.variables[vname]
            dims = var.dimensions
            if dims[-2:] != (lat_name, lon_name):
                raise ValueError(f"{vname}: expected trailing dims ({lat_name}, {lon_name}), "
                                 f"got {dims}")
            tw = time.time()
            fill = getattr(var, '_FillValue', None)
            out_dtype = var.dtype if var.dtype.kind == 'f' else np.float32
            if time_name in dims:
                chunks = (1, tgt_lat.size, tgt_lon.size)
            else:
                chunks = (tgt_lat.size, tgt_lon.size)
            v = out.createVariable(vname, out_dtype, dims, zlib=True, complevel=complevel,
                                   shuffle=True, chunksizes=chunks, fill_value=fill)
            _copy_attrs(var, v)
            v.regrid_method = 'conservative area-weighted (see global attributes)'
            if vname == 'lsm':
                v.regrid_note = ('area-weighted mean of the source land-sea mask, i.e. a '
                                 'LAND FRACTION in [0, 1], not a binary mask; odem.py '
                                 'thresholds it at 0.5')
            if vname == 'slt':
                v.regrid_note = ('categorical soil-type codes were area-averaged and are '
                                 'not meaningful as classes; odem.py does not use slt')

            if time_name in dims:
                for t_start in range(0, nt, chunk):
                    t_end = min(t_start + chunk, nt)
                    block = np.ma.filled(var[t_start:t_end], np.nan).astype(np.float64)
                    res = rg(block, fill=np.nan)
                    v[t_start:t_end] = res.astype(out_dtype)
            else:
                block = np.ma.filled(var[:], np.nan).astype(np.float64)
                v[:] = rg(block, fill=np.nan).astype(out_dtype)
            print(f"  {vname:8s} {time.time() - tw:7.1f} s")
        out.sync()
    finally:
        out.close()
        src.close()

    size_mb = out_path.stat().st_size / 1e6
    print(f"Done in {time.time() - t0:.1f} s; {out_path} ({size_mb:.1f} MB)")
    return out_path


def _copy_attrs(src_var, dst_var, skip=()):
    for a in src_var.ncattrs():
        if a == '_FillValue' or a in skip:
            continue
        dst_var.setncattr(a, src_var.getncattr(a))


def main():
    p = argparse.ArgumentParser(
        description='Conservative regrid of an ERA5 monthly file onto a MERRA-2 grid')
    p.add_argument('--src', required=True, help='ERA5 monthly NetCDF (era5_dust_YYYYMM.nc)')
    p.add_argument('--target-grid', required=True,
                   help='NetCDF whose lat/lon define the target grid (MERRA-2 monthly file)')
    p.add_argument('--out', required=True, help='Output NetCDF path')
    p.add_argument('--steps', type=int, default=None,
                   help='Only the first N timesteps (for tests); default all')
    p.add_argument('--chunk', type=int, default=8,
                   help='Timesteps processed per block (default 8)')
    args = p.parse_args()
    regrid_file(args.src, args.target_grid, args.out, steps=args.steps, chunk=args.chunk)


if __name__ == '__main__':
    main()
