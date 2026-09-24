#!/usr/bin/env python
"""Convert monthly MODIS BRDF GeoTIFFs (GEE export: bands albedo, shadow,
omega_n, f_iso) into one NetCDF file that odem.py reads for the albedo-based
drag partition scheme, so the model itself needs no GeoTIFF reader.

Values are copied as float32 without change. Cell-centre coordinates are
computed from the GeoTIFF affine transform exactly as the development code
did (lon = c + (j + 0.5) a, lat = f + (i + 0.5) e), so a model run from the
NetCDF reproduces a run from the GeoTIFFs bitwise.

Usage:
  python scripts/brdf_tif_to_nc.py --tif-dir E:/.../modis_brdf --year 2006 \
      --out E:/.../modis_brdf/modis_brdf_2006.nc
Needs rasterio (conversion only, not a model dependency).
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import netCDF4 as nc
import rasterio

BANDS = ['albedo', 'shadow', 'omega_n', 'f_iso']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tif-dir', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    tif_dir = Path(a.tif_dir)
    files = [tif_dir / f'brdf_{a.year}{m:02d}.tif' for m in range(1, 13)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'missing: {missing}')
    with rasterio.open(files[0]) as ds:
        tr = ds.transform
        nrows, ncols = ds.height, ds.width
        lon = np.array([tr.c + (j + 0.5) * tr.a for j in range(ncols)])
        lat = np.array([tr.f + (i + 0.5) * tr.e for i in range(nrows)])
        assert list(ds.descriptions) == BANDS, ds.descriptions
    with nc.Dataset(a.out, 'w', format='NETCDF4') as out:
        out.createDimension('month', 12)
        out.createDimension('lat', nrows)
        out.createDimension('lon', ncols)
        v = out.createVariable('month', 'i4', ('month',)); v[:] = np.arange(1, 13); v.units = f'month of {a.year}'
        v = out.createVariable('lat', 'f8', ('lat',)); v[:] = lat; v.units = 'degrees_north'; v.note = 'cell centre, from GeoTIFF transform'
        v = out.createVariable('lon', 'f8', ('lon',)); v[:] = lon; v.units = 'degrees_east'
        vars_ = {}
        for b in BANDS:
            vars_[b] = out.createVariable(b, 'f4', ('month', 'lat', 'lon'), zlib=True, complevel=1)
        for mi, f in enumerate(files):
            with rasterio.open(f) as ds:
                for bi, b in enumerate(BANDS, start=1):
                    vars_[b][mi, :, :] = ds.read(bi)
            print(f'  {f.name}')
        vars_['omega_n'].long_name = 'normalised shadow omega_n = (1 - albedo) / f_iso, MODIS MCD43 BRDF (GEE export)'
        vars_['albedo'].long_name = 'MODIS MCD43 black-sky albedo'
        vars_['shadow'].long_name = 'shadow fraction'
        vars_['f_iso'].long_name = 'isotropic BRDF kernel weight'
        out.title = f'MODIS BRDF omega_n and components, monthly {a.year}, for ODEM albedo-based drag partition'
        out.source = f'GeoTIFF files brdf_{a.year}MM.tif exported from Google Earth Engine (Chappell and Webb 2016 method)'
        out.transform = str(tuple(tr))
        out.history = f'Created {dt.datetime.now().isoformat()} by scripts/brdf_tif_to_nc.py'
    print('wrote', a.out)


if __name__ == '__main__':
    main()
