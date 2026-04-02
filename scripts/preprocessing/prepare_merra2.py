#!/usr/bin/env python3
"""
prepare_merra2.py — Merge daily MERRA-2 files into monthly NetCDF for ODEM
===========================================================================
Reads daily MERRA-2 files from the flx/ and lnd/ subdirectories (downloaded
by download_merra2.py) and merges them into monthly merra2_dust_YYYYMM.nc
files that odem.py --merra2 can read directly.

Variables extracted
-------------------
From M2T1NXFLX (flx/):  USTAR, RHOA, PBLH
From M2T1NXLND (lnd/):  SFMC, SNODP

Output file variables
---------------------
USTAR   : friction velocity [m/s]          (time, lat, lon)
RHOA    : air density [kg/m³]              (time, lat, lon)
PBLH    : boundary layer height [m]        (time, lat, lon)  — optional
SFMC    : surface soil moisture [m³/m³]    (time, lat, lon)
SNODP   : snow depth [m, actual]           (time, lat, lon)

Grid: 0.5° lat × 0.625° lon  (361 × 576)
Time: 1-hourly, concatenated across all days in month

Usage
-----
  # Single month
  python prepare_merra2.py \\
      --merra2-dir /path/to/merra2_data \\
      --output-dir /path/to/merra2_monthly \\
      --year 2006 --month 1

  # Full year
  python prepare_merra2.py \\
      --merra2-dir /path/to/merra2_data \\
      --output-dir /path/to/merra2_monthly \\
      --year 2006

Scientific notes
----------------
- SNODP: actual snow depth (m), not SWE. ODEM uses threshold 0.01 m (any detectable snow).
- SFMC: volumetric soil moisture (m³/m³) at surface layer (~0-5 cm).
  Equivalent to ERA5 swvl1; used with SoilGrids bulk density for gravimetric water content.
- RHOA: air density at lowest model level (~10 m). Provided directly by MERRA-2 land model;
  ODEM uses it instead of computing from T, q, p (Magnus formula path for ERA5).
- PBLH: planetary boundary layer height [m]. Used for Comola et al. (2019) intermittency.
  Optional — ODEM skips intermittency if PBLH missing.

Authors: Metin Baykara
"""

import os
import sys
import argparse
import logging
import calendar
from pathlib import Path
from datetime import date

import numpy as np
import netCDF4 as nc4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------
FLX_VARS = ['USTAR', 'RHOA', 'PBLH', 'DISPH']  # from M2T1NXFLX
LND_VARS = ['SFMC', 'SNODP']                   # from M2T1NXLND

FILL_VALUE = 1.0e15   # MERRA-2 fill value for missing/ocean cells


def find_daily_file(merra2_dir, collection_short, date_str):
    """
    Find MERRA-2 daily file for given collection and date.

    Expected naming: MERRA2_<stream>.tavg1_2d_<coll>_Nx.<YYYYMMDD>.nc4
    collection_short: 'flx' or 'lnd'
    date_str: 'YYYYMMDD'
    """
    coll_full = f'tavg1_2d_{collection_short}_Nx'
    sub = Path(merra2_dir) / collection_short
    patterns = [
        f'MERRA2_*.{coll_full}.{date_str}.nc4',
        f'MERRA2_*.{coll_full}.{date_str}.*.nc4',
        f'MERRA2_*.{coll_full}.{date_str}.SUB.nc4',
    ]
    for pat in patterns:
        matches = list(sub.glob(pat))
        if matches:
            return str(matches[0])
    return None


def read_daily(filepath, varnames):
    """
    Read variables from one daily MERRA-2 file.

    Returns dict: varname → ndarray shape (24, 361, 576) or (nt, nlat, nlon).
    Ocean/missing cells retain fill value (~1e15).
    """
    result = {}
    with nc4.Dataset(filepath, 'r') as ds:
        for v in varnames:
            if v not in ds.variables:
                log.warning(f"  {v} not in {Path(filepath).name}")
                continue
            data = np.array(ds.variables[v][:], dtype=np.float32)
            # Replace obvious fill values with NaN
            data[np.abs(data) > 1e10] = np.nan
            result[v] = data   # shape: (time, lat, lon)
    return result


def merge_month(merra2_dir, year, month, output_dir):
    """
    Merge all daily MERRA-2 files for a given year/month into one NetCDF.

    Output: <output_dir>/merra2_dust_YYYYMM.nc
    """
    ndays = calendar.monthrange(year, month)[1]
    out_file = Path(output_dir) / f'merra2_dust_{year}{month:02d}.nc'
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Merging {year}-{month:02d} ({ndays} days) → {out_file}")

    # Collect daily arrays
    all_flx = {v: [] for v in FLX_VARS}
    all_lnd = {v: [] for v in LND_VARS}
    times_all = []   # hours since 1901-01-01 (CF convention)
    ref_date = date(1901, 1, 1)
    n_found = 0

    for day in range(1, ndays + 1):
        date_str = f'{year}{month:02d}{day:02d}'
        flx_path = find_daily_file(merra2_dir, 'flx', date_str)
        lnd_path = find_daily_file(merra2_dir, 'lnd', date_str)

        if flx_path is None or lnd_path is None:
            log.warning(f"  {date_str}: missing {'flx' if flx_path is None else 'lnd'} — skipping")
            continue

        log.info(f"  {date_str}: {Path(flx_path).name}")

        flx = read_daily(flx_path, FLX_VARS)
        lnd = read_daily(lnd_path, LND_VARS)

        # Record times: hours since ref_date for each of the 24 hours
        d0 = date(year, month, day)
        hours_since = (d0 - ref_date).days * 24
        nt_day = flx.get('USTAR', lnd.get('SFMC')).shape[0]
        times_all.extend([hours_since + h for h in range(nt_day)])

        for v in FLX_VARS:
            if v in flx:
                all_flx[v].append(flx[v])
        for v in LND_VARS:
            if v in lnd:
                all_lnd[v].append(lnd[v])

        n_found += 1

    if n_found == 0:
        log.error(f"No files found for {year}-{month:02d}. Skipping.")
        return None

    log.info(f"  {n_found}/{ndays} days found, {len(times_all)} timesteps total")

    # Concatenate along time axis
    merged = {}
    for v in FLX_VARS:
        if all_flx[v]:
            merged[v] = np.concatenate(all_flx[v], axis=0)  # (nt, lat, lon)
    for v in LND_VARS:
        if all_lnd[v]:
            merged[v] = np.concatenate(all_lnd[v], axis=0)

    # Get grid from first available array
    first_arr = next(iter(merged.values()))
    nt, nlat, nlon = first_arr.shape

    # Get lat/lon from first file
    first_flx = find_daily_file(merra2_dir, 'flx', f'{year}{month:02d}01')
    with nc4.Dataset(first_flx, 'r') as ds:
        lat = np.array(ds.variables['lat'][:])
        lon = np.array(ds.variables['lon'][:])

    # Write output
    _write_monthly(out_file, merged, lat, lon, np.array(times_all, dtype=np.float64),
                   year, month, n_found)
    return str(out_file)


def _write_monthly(out_file, merged, lat, lon, times, year, month, n_days_found):
    """Write merged monthly NetCDF file."""
    nt = len(times)
    nlat = len(lat)
    nlon = len(lon)

    # Variable metadata: (long_name, units)
    VAR_META = {
        'USTAR': ('Friction velocity',                'm s-1'),
        'RHOA':  ('Air density',                      'kg m-3'),
        'PBLH':  ('Planetary boundary layer height',  'm'),
        'DISPH': ('Displacement height (land mask)',   'm'),
        'SFMC':  ('Surface soil moisture',            'm3 m-3'),
        'SNODP': ('Snow depth',                       'm'),
    }

    with nc4.Dataset(str(out_file), 'w', format='NETCDF4') as ds:
        # Dimensions
        ds.createDimension('time', nt)
        ds.createDimension('lat',  nlat)
        ds.createDimension('lon',  nlon)

        # Coordinate variables
        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units = 'hours since 1901-01-01 00:00:00'
        tv.calendar = 'standard'
        tv[:] = times

        latv = ds.createVariable('lat', 'f4', ('lat',))
        latv.units = 'degrees_north'
        latv[:] = lat

        lonv = ds.createVariable('lon', 'f4', ('lon',))
        lonv.units = 'degrees_east'
        lonv[:] = lon

        # Data variables
        for varname, arr in merged.items():
            long_name, units = VAR_META.get(varname, (varname, ''))
            v = ds.createVariable(varname, 'f4', ('time', 'lat', 'lon'),
                                  fill_value=FILL_VALUE,
                                  zlib=True, complevel=4)
            v.long_name = long_name
            v.units = units
            # Replace NaN → fill_value for NetCDF storage
            arr_out = np.where(np.isnan(arr), FILL_VALUE, arr).astype(np.float32)
            v[:] = arr_out

        # Global attributes
        ds.title = f'MERRA-2 dust emission inputs {year}-{month:02d}'
        ds.source = ('M2T1NXFLX.5.12.4 (USTAR, RHOA, PBLH), '
                     'M2T1NXLND.5.12.4 (SFMC, SNODP)')
        ds.resolution = '0.5 deg lat x 0.625 deg lon'
        ds.days_found = n_days_found
        ds.timesteps = nt
        ds.prepared_by = 'prepare_merra2.py'

    log.info(f"  Written: {out_file} ({nt} timesteps, "
             f"{os.path.getsize(str(out_file)) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Merge daily MERRA-2 files into monthly NetCDF for ODEM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single month
  python prepare_merra2.py \\
      --merra2-dir /path/to/merra2_data \\
      --output-dir /path/to/merra2_monthly \\
      --year 2006 --month 1

  # Full year
  python prepare_merra2.py \\
      --merra2-dir /path/to/merra2_data \\
      --output-dir /path/to/merra2_monthly \\
      --year 2006
""")
    p.add_argument('--merra2-dir', required=True,
                   help='Root MERRA-2 data directory (contains flx/, lnd/ subdirs)')
    p.add_argument('--output-dir', required=True,
                   help='Output directory for merra2_dust_YYYYMM.nc files')
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--month', type=int, default=None,
                   help='Month 1-12 (omit for full year)')

    args = p.parse_args()

    months = [args.month] if args.month else list(range(1, 13))

    for m in months:
        out = merge_month(args.merra2_dir, args.year, m, args.output_dir)
        if out:
            log.info(f"  → {out}")


if __name__ == '__main__':
    main()
