#!/usr/bin/env python3
"""
Static Dataset Preparation for ODEM
====================================
Prepares the time-invariant input datasets needed by odem.py.

The R code uses pre-processed .RData files. This script recreates those
datasets from publicly available sources and saves them as NetCDF.

Required static datasets:
    1. Clay fraction        → clay_fao.nc  (from CESM/FAO or SoilGrids)
    2. Land cover fractions → glcnmo_fractions.nc  (from GLCNMO)
    3. Roughness length     → prigent_z0.nc  (from Prigent et al. 2005)
    4. Land/ocean mask      → landfilt.nc
    5. Porosity             → from MERRA2 const file (downloaded separately)

Data sources:
    - FAO clay: CESM input data — https://svn-ccsm-inputdata.cgd.ucar.edu/
    - SoilGrids: https://soilgrids.org/ (250m, needs aggregation to 0.5°×0.625°)
    - GLCNMO: https://globalmaps.github.io/glcnmo.html (v3, 2013)
    - Prigent 2005: roughness from satellite, available via authors
    - Land mask: derived from MERRA2 land fraction or soil data

Authors: Metin Baykara
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import netCDF4 as nc4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# MERRA2 grid definition (0.5° × 0.625°)
# ===========================================================================
MERRA2_LON = np.arange(-180.0, 180.0, 0.625)    # 576 points
MERRA2_LAT = np.arange(-90.0, 90.25, 0.5)       # 361 points


def create_merra2_grid():
    """Return the native MERRA2 0.5° × 0.625° grid."""
    return MERRA2_LON, MERRA2_LAT


# ===========================================================================
# Option A: Convert existing .RData files to NetCDF (if R is available)
# ===========================================================================

def convert_rdata_to_netcdf(rdata_dir, output_dir):
    """
    Convert the R .RData files from the Leung 2023 code to NetCDF.

    Requires rpy2 (pip install rpy2) or calls R via subprocess.
    This is the easiest path if you have the original .RData files.
    """
    rdata_dir = Path(rdata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # R script to convert .RData → NetCDF
    r_script = f"""
    library(ncdf4)

    setwd("{rdata_dir.as_posix()}")

    # Grid
    lon = seq(-180, 179.375, by=0.625)
    lat = seq(-90, 90, by=0.5)

    save_nc = function(data, varname, longname, units, filename) {{
        dim_lon = ncdim_def("lon", "degrees_east", lon)
        dim_lat = ncdim_def("lat", "degrees_north", lat)
        var = ncvar_def(varname, units, list(dim_lon, dim_lat), missval=-9999)
        nc = nc_create(filename, var)
        ncvar_put(nc, var, data)
        ncatt_put(nc, 0, "title", longname)
        nc_close(nc)
        cat("Saved:", filename, "\\n")
    }}

    # 1. Clay fraction (FAO)
    if (file.exists("f_clay_FAO_05x06.RData")) {{
        load("f_clay_FAO_05x06.RData")
        save_nc(f_clay.FAO.05x06, "f_clay", "FAO clay mass fraction", "fraction",
                "{(output_dir / 'clay_fao.nc').as_posix()}")
    }}

    # 2. Clay fraction (SoilGrids)
    if (file.exists("f_clay_SoilGrids_05x06.RData")) {{
        load("f_clay_SoilGrids_05x06.RData")
        save_nc(f_clay.SG.05x06, "f_clay", "SoilGrids clay mass fraction", "fraction",
                "{(output_dir / 'clay_sg.nc').as_posix()}")
    }}

    # 3. GLCNMO land cover fractions
    if (file.exists("GLCNMO_frc_area_05x0625.RData")) {{
        load("GLCNMO_frc_area_05x0625.RData")
        dim_lon = ncdim_def("lon", "degrees_east", lon)
        dim_lat = ncdim_def("lat", "degrees_north", lat)
        v1 = ncvar_def("frc_rock", "fraction", list(dim_lon, dim_lat), missval=-9999)
        v2 = ncvar_def("frc_veg", "fraction", list(dim_lon, dim_lat), missval=-9999)
        v3 = ncvar_def("frc_mix", "fraction", list(dim_lon, dim_lat), missval=-9999)
        nc = nc_create("{(output_dir / 'glcnmo_fractions.nc').as_posix()}", list(v1, v2, v3))
        ncvar_put(nc, v1, frc.r.M2)
        ncvar_put(nc, v2, frc.veg.M2)
        ncvar_put(nc, v3, frc.mix.M2)
        nc_close(nc)
        cat("Saved: glcnmo_fractions.nc\\n")
    }}

    # 4. Land filter
    if (file.exists("landfilt_05x0625.RData")) {{
        load("landfilt_05x0625.RData")
        save_nc(landfilt.M2, "landfilt", "Land filter (1=land, 0=ocean)", "1",
                "{(output_dir / 'landfilt.nc').as_posix()}")
    }}

    # 5. Prigent 2005 roughness
    if (file.exists("Pr05_z0.min_05x0625.RData")) {{
        load("Pr05_z0.min_05x0625.RData")
        save_nc(z0.M2, "z0", "Prigent 2005 surface roughness", "cm",
                "{(output_dir / 'prigent_z0.nc').as_posix()}")
    }}

    cat("\\nConversion complete.\\n")
    """

    r_script_path = output_dir / "_convert_rdata.R"
    r_script_path.write_text(r_script)

    log.info(f"R conversion script written to: {r_script_path}")
    log.info("Run it with: Rscript _convert_rdata.R")

    # Try to run R automatically
    import subprocess
    try:
        result = subprocess.run(
            ['Rscript', str(r_script_path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log.info("R conversion completed successfully.")
            log.info(result.stdout)
        else:
            log.warning(f"R conversion failed: {result.stderr}")
            log.info("You can run it manually: Rscript " + str(r_script_path))
    except FileNotFoundError:
        log.info("R not found in PATH. Run the script manually:")
        log.info(f"  Rscript {r_script_path}")
    except subprocess.TimeoutExpired:
        log.warning("R conversion timed out.")


# ===========================================================================
# Option B: Create static datasets from scratch (no R needed)
# ===========================================================================

def create_landfilt_from_merra2(const_file, output_file, indi=None, indj=None):
    """
    Create a land filter from MERRA2 constant fields.

    Uses porosity: where porosity > 0 → land; where porosity = 0 or fill → ocean.
    """
    log.info(f"Creating land filter from {const_file}...")

    with nc4.Dataset(const_file, 'r') as ds:
        poros = ds.variables['poros'][:]
        if poros.ndim == 3:
            poros = poros[0, :, :]
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]

    # Transpose to (lon, lat) if needed
    if poros.shape == (len(lat), len(lon)):
        poros = poros.T

    # Land where porosity is valid and > 0
    landfilt = np.where((poros > 0) & (~np.isnan(poros)), 1.0, 0.0)

    # Save
    _save_2d_nc(output_file, lon, lat, landfilt, 'landfilt',
                'Land filter (1=land, 0=ocean/water)', '1')
    log.info(f"Saved: {output_file}")
    return landfilt


def create_porosity_nc(const_file, output_file):
    """Extract porosity from MERRA2 constant file and save as simple NetCDF."""
    log.info(f"Extracting porosity from {const_file}...")

    with nc4.Dataset(const_file, 'r') as ds:
        poros = ds.variables['poros'][:]
        if poros.ndim == 3:
            poros = poros[0, :, :]
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]

    if poros.shape == (len(lat), len(lon)):
        poros = poros.T

    _save_2d_nc(output_file, lon, lat, poros, 'poros',
                'Volumetric soil porosity', 'm3/m3')
    log.info(f"Saved: {output_file}")


def download_and_process_soilgrids_clay(output_file, resolution='0.5x0.625'):
    """
    Download SoilGrids clay fraction and regrid to MERRA2 grid.

    SoilGrids provides clay content at 250m resolution.
    We use their pre-aggregated 5km product and regrid to 0.5°×0.625°.

    API: https://rest.isric.org/soilgrids/v2.0/properties/query
    Bulk download: https://files.isric.org/soilgrids/latest/data/
    """
    log.info("SoilGrids clay download requires manual steps:")
    log.info("  1. Go to https://soilgrids.org/")
    log.info("  2. Download clay content (0-5cm) at desired resolution")
    log.info("  3. Or use the ISRIC WCS service:")
    log.info("     https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean/")
    log.info("")
    log.info("  Alternative: Use the pre-processed file from the R code if available.")
    log.info(f"  Expected output: {output_file}")


def create_dummy_static_data(output_dir):
    """
    Create dummy/placeholder static datasets for testing.

    These use reasonable global defaults — NOT for production use.
    Replace with real data before validation runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lon, lat = create_merra2_grid()
    nlon, nlat = len(lon), len(lat)

    log.info("Creating DUMMY static datasets for testing...")
    log.info("WARNING: These are placeholders — replace with real data!")

    # 1. Clay fraction (global average ~0.15)
    f_clay = np.full((nlon, nlat), 0.15)
    # Reduce clay in known sandy regions (rough approximation)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)
    sahara_mask = (lat_mesh > 15) & (lat_mesh < 35) & (lon_mesh > -15) & (lon_mesh < 40)
    f_clay[sahara_mask] = 0.05
    _save_2d_nc(str(output_dir / 'clay_fao.nc'), lon, lat, f_clay,
                'f_clay', 'Clay mass fraction (DUMMY)', 'fraction')

    # 2. Land cover fractions (simplified: desert=rock, rest=veg)
    frc_rock = np.zeros((nlon, nlat))
    frc_veg = np.ones((nlon, nlat))
    frc_mix = np.zeros((nlon, nlat))
    # Set known desert regions as rock-dominated
    desert_mask = (
        sahara_mask |
        ((lat_mesh > 25) & (lat_mesh < 45) & (lon_mesh > 45) & (lon_mesh < 80)) |  # Middle East/Central Asia
        ((lat_mesh > -35) & (lat_mesh < -20) & (lon_mesh > 15) & (lon_mesh < 30))    # Namibia/Kalahari
    )
    frc_rock[desert_mask] = 0.7
    frc_veg[desert_mask] = 0.2
    frc_mix[desert_mask] = 0.1

    with nc4.Dataset(str(output_dir / 'glcnmo_fractions.nc'), 'w', format='NETCDF4') as ds:
        ds.createDimension('lon', nlon)
        ds.createDimension('lat', nlat)
        lon_v = ds.createVariable('lon', 'f8', ('lon',)); lon_v[:] = lon
        lat_v = ds.createVariable('lat', 'f8', ('lat',)); lat_v[:] = lat
        for name, data, desc in [
            ('frc_rock', frc_rock, 'Rock area fraction (DUMMY)'),
            ('frc_veg', frc_veg, 'Vegetation area fraction (DUMMY)'),
            ('frc_mix', frc_mix, 'Mixed area fraction (DUMMY)'),
        ]:
            v = ds.createVariable(name, 'f4', ('lon', 'lat'), zlib=True)
            v[:] = data
            v.long_name = desc
        ds.title = "DUMMY GLCNMO land cover fractions — replace with real data"

    # 3. Roughness (Prigent 2005 — approximate with z0=0.01 cm for flat desert)
    z0_cm = np.full((nlon, nlat), 1.0)  # 1 cm default
    z0_cm[desert_mask] = 0.01  # Very smooth desert
    _save_2d_nc(str(output_dir / 'prigent_z0.nc'), lon, lat, z0_cm,
                'z0', 'Surface roughness (DUMMY)', 'cm')

    # 4. Land filter (ocean=0, land=1)
    # Simple: land where |lat| < 80 and not major oceans
    landfilt = np.ones((nlon, nlat))
    # Crude ocean mask
    landfilt[:, lat < -60] = 0  # Antarctica
    landfilt[:, lat > 85] = 0   # Arctic ocean
    _save_2d_nc(str(output_dir / 'landfilt.nc'), lon, lat, landfilt,
                'landfilt', 'Land filter (DUMMY)', '1')

    # 5. Porosity (global average ~0.45)
    poros = np.full((nlon, nlat), 0.45)
    poros[desert_mask] = 0.35  # Sandy soil = lower porosity
    _save_2d_nc(str(output_dir / 'merra2_const.nc'), lon, lat, poros,
                'poros', 'Volumetric soil porosity (DUMMY)', 'm3/m3')

    log.info(f"DUMMY static datasets created in: {output_dir}")
    log.info("Files: clay_fao.nc, glcnmo_fractions.nc, prigent_z0.nc, landfilt.nc, merra2_const.nc")


def _save_2d_nc(filepath, lon, lat, data, varname, longname, units):
    """Helper: save a 2D (lon, lat) field to NetCDF."""
    with nc4.Dataset(filepath, 'w', format='NETCDF4') as ds:
        ds.createDimension('lon', len(lon))
        ds.createDimension('lat', len(lat))

        lon_v = ds.createVariable('lon', 'f8', ('lon',))
        lon_v[:] = lon
        lon_v.units = 'degrees_east'

        lat_v = ds.createVariable('lat', 'f8', ('lat',))
        lat_v[:] = lat
        lat_v.units = 'degrees_north'

        v = ds.createVariable(varname, 'f4', ('lon', 'lat'), zlib=True, complevel=4)
        v[:] = data
        v.long_name = longname
        v.units = units


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Prepare static datasets for ODEM"
    )
    sub = p.add_subparsers(dest='command')

    # Convert R .RData files
    conv = sub.add_parser('convert-rdata',
                          help='Convert .RData files to NetCDF (requires R)')
    conv.add_argument('--rdata-dir', required=True,
                      help='Directory containing .RData files')
    conv.add_argument('--output-dir', required=True,
                      help='Output directory for NetCDF files')

    # Create from MERRA2 const
    merra = sub.add_parser('from-merra2',
                           help='Create landfilt and porosity from MERRA2 const file')
    merra.add_argument('--const-file', required=True,
                       help='MERRA2 const_2d_lnd_Nx file path')
    merra.add_argument('--output-dir', required=True,
                       help='Output directory')

    # Dummy data for testing
    dummy = sub.add_parser('dummy',
                           help='Create DUMMY static data for testing (not for production!)')
    dummy.add_argument('--output-dir', required=True,
                       help='Output directory')

    args = p.parse_args()

    if args.command == 'convert-rdata':
        convert_rdata_to_netcdf(args.rdata_dir, args.output_dir)
    elif args.command == 'from-merra2':
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        create_landfilt_from_merra2(args.const_file, str(out / 'landfilt.nc'))
        create_porosity_nc(args.const_file, str(out / 'merra2_const.nc'))
    elif args.command == 'dummy':
        create_dummy_static_data(args.output_dir)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
