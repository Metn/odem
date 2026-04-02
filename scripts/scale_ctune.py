#!/usr/bin/env python3
"""
Scale ODEM output files to a different C_tune value.

C_tune is a linear multiplier in the Kok (2014) emission equation, so changing
C_tune simply scales all emission fields by (new / old). Spatial patterns,
temporal variability, and reanalysis ratios are unchanged.

Usage:
    python scale_ctune.py --input-dir /path/to/output_2006 --output-dir /path/to/output_2006_ctune020
    python scale_ctune.py --input-dir /path/to/output_2006 --output-dir /path/to/output_2006_ctune020 --scale-factor 0.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Variables that contain emission flux and must be scaled
EMISSION_VARS_MONTHLY = ['F_d']
EMISSION_VARS_ANNUAL = ['F_d_mean', 'F_d_max']
SKIP_FILES = ['odem_diag_mean.nc']


def scale_file(src: Path, dst: Path, scale: float):
    """Read a NetCDF, scale emission variables, write to dst."""
    ds = xr.open_dataset(src)

    # Determine which variables to scale
    if 'F_d' in ds.data_vars:
        to_scale = [v for v in EMISSION_VARS_MONTHLY if v in ds.data_vars]
    else:
        to_scale = [v for v in EMISSION_VARS_ANNUAL if v in ds.data_vars]

    if not to_scale:
        print(f"  SKIP (no emission vars): {src.name}")
        ds.close()
        return

    # Compute original total for summary (from annual file only)
    orig_total = None
    if 'F_d_mean' in to_scale:
        # Quick global total estimate (kg/m²/s → Tg/yr)
        lat = ds['lat'].values
        lon = ds['lon'].values
        dlat = abs(np.diff(lat).mean()) * np.pi / 180
        dlon = abs(np.diff(lon).mean()) * np.pi / 180
        R = 6.371e6
        cos_lat = np.cos(np.deg2rad(lat))
        area = R**2 * dlat * dlon * cos_lat[:, np.newaxis] * np.ones((1, len(lon)))
        orig_total = float(np.nansum(ds['F_d_mean'].values * area)) * 3.1536e7 / 1e9

    # Scale
    for v in to_scale:
        ds[v] = ds[v] * scale

    # Preserve encoding
    encoding = {}
    for v in ds.data_vars:
        encoding[v] = {'dtype': 'float32', 'zlib': True, 'complevel': 4}

    # Add provenance attribute
    ds.attrs['C_tune_scaling'] = (
        f"Scaled from C_tune=0.05 to C_tune=0.020 (factor {scale}). "
        f"Spatial patterns and reanalysis ratios are unchanged."
    )

    ds.to_netcdf(dst, encoding=encoding)
    ds.close()

    if orig_total is not None:
        print(f"  {src.name} -> {dst.name}  "
              f"(original: {orig_total:.0f} Tg/yr, scaled: {orig_total * scale:.0f} Tg/yr)")
    else:
        print(f"  {src.name} -> {dst.name}  (scaled by {scale})")


def main():
    p = argparse.ArgumentParser(description="Scale ODEM outputs to a different C_tune")
    p.add_argument('--input-dir', required=True, help='Directory with original odem_*.nc files')
    p.add_argument('--output-dir', required=True, help='Directory for scaled output files')
    p.add_argument('--scale-factor', type=float, default=0.4,
                   help='Multiplicative scale factor (default: 0.4 = C_tune 0.020/0.05)')
    args = p.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)

    if not indir.exists():
        print(f"ERROR: input directory does not exist: {indir}")
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(indir.glob('odem_*.nc'))
    if not nc_files:
        print(f"ERROR: no odem_*.nc files in {indir}")
        sys.exit(1)

    print(f"Input:  {indir}")
    print(f"Output: {outdir}")
    print(f"Scale:  {args.scale_factor} (C_tune 0.05 -> {0.05 * args.scale_factor:.3f})")
    print(f"Files:  {len(nc_files)}")
    print()

    for f in nc_files:
        if f.name in SKIP_FILES:
            print(f"  SKIP (diagnostics): {f.name}")
            continue
        scale_file(f, outdir / f.name, args.scale_factor)

    print("\nDone.")


if __name__ == '__main__':
    main()
