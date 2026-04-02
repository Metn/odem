#!/usr/bin/env python3
"""
ODEM — Offline Dust Emission Model
===================================

Process-based dust emission model implementing the scheme described in:
  - Kok et al. (2014) ACP — Base emission equation
  - Leung, Kok et al. (2023) ACP — Part I: Theory & evaluation
  - Leung, Kok et al. (2024) ACP — Part II: CESM2 evaluation

Physics included:
  - Kok et al. (2014) brittle fragmentation dust emission equation
  - Shao & Lu (2000) fluid threshold with soil particle size dependence
  - Impact threshold hysteresis (u*_it = 0.82 × u*_ft0)
  - Drag partition: Rock (MB95/Prigent z0a) + Vegetation (Okin 2008 gap-size)
  - Fécan et al. (1999) soil moisture correction
  - Comola et al. (2019) turbulent intermittency
  - Spatially varying median soil diameter from clay+silt fractions
  - Fragmentation exponent capped at 3

Input datasets:
  - ERA5 surface fields (meteorology)
  - Prigent et al. (2005) aeolian roughness z0a (ERS scatterometer)
  - MODIS Collection 6.1 LAI (Hamburg/ICDC regridded 0.5°, 8-day)
  - SoilGrids v2.0 clay, silt, bulk density

Author: Metin Baykara
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import netCDF4 as nc4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Physical constants — Leung et al. (2023) Table 1
# ===========================================================================

CONST = {
    # Fundamental
    'g':          9.81,       # m/s², gravitational acceleration
    'k_vk':       0.4,        # von Kármán constant

    # Soil particle properties
    'rho_p':      2650.0,     # kg/m³, soil particle density
    'rho_water':  1000.0,     # kg/m³, water density

    # Shao & Lu (2000) threshold parameters
    'A_SL':       0.0123,     # aerodynamic force coefficient
    'gamma_SL':   1.65e-4,    # kg/s², interparticle cohesion

    # Median soil diameter
    'D_p_arid':   127e-6,     # m, median particle diameter in arid regions
    'Psi_0':      7.8e-6,     # m, intercept for non-arid D_p regression
    'Psi_1':      124e-6,     # m, slope for non-arid D_p regression

    # Kok et al. (2014) emission equation
    'C_tune':     0.05,       # global tuning constant
    'C_d0':       4.4e-5,     # dust emission coefficient baseline
    'C_e':        2.0,        # dust emission coefficient exponent
    'C_alpha':    2.7,        # fragmentation exponent coefficient
    'ustar_st0':  0.16,       # m/s, standardized fluid threshold reference
    'kappa_cap':  3.0,        # maximum fragmentation exponent

    # Impact threshold
    'B_it':       0.82,       # impact/fluid threshold ratio

    # Air density reference
    'rho_a0':     1.225,      # kg/m³, standard air density

    # Rock drag partition (Marticorena & Bergametti 1995)
    'b1':         0.7,        # MB95 parameter
    'b2':         0.8,        # MB95 parameter
    'X':          10.0,       # m, inter-obstacle distance

    # Vegetation drag partition (Okin 2008)
    'f0':         0.32,       # friction ratio immediately behind obstacle
    'c_okin':     4.8,        # e-folding distance / vegetation height

    # Vegetation threshold
    'LAI_thr':    1.0,        # LAI threshold for bare soil fraction

    # Soil moisture (Fécan et al. 1999)
    # NOTE: a_moist=1.0 is standard Fécan. Previously set to 2.0 as
    # ad-hoc correction for ERA5 swvl1 ≠ surface gravimetric moisture.
    # Reverted to 1.0 for reproducibility. ERA5 swvl1 → GWC conversion
    # via bulk density is the proper approach (already implemented).
    'a_moist':    1.0,        # soil moisture tuning (1.0 = standard Fécan)

}


# ===========================================================================
# MERRA-2 variable mapping
# ===========================================================================
# Maps ODEM internal field names → MERRA-2 variable names in merged monthly files.
# Merged file format: merra2_dust_YYYYMM.nc — dimensions (time, lat, lon)
# Resolution: 0.5° lat × 0.625° lon (361 × 576)
# Sources: M2T1NXFLX.5.12.4 (flx), M2T1NXLND.5.12.4 (lnd), M2C0NXLND.5.12.4 (const)
MERRA2_VARS = {
    'zust':  'USTAR',    # friction velocity [m/s]         — M2T1NXFLX
    'rhoa':  'RHOA',     # air density [kg/m³]             — M2T1NXFLX (direct, skip compute)
    'blh':   'PBLH',     # boundary layer height [m]       — M2T1NXFLX (optional)
    'swvl1': 'SFMC',     # surface soil moisture [m³/m³]   — M2T1NXLND
    'sd':    'SNODP',    # snow depth [m, actual]          — M2T1NXLND (NOT SHLAND=heat flux)
}
# Land mask: derived in _load_merra2_grid() from SFMC validity (land cells have finite SFMC).
# Snow note: MERRA-2 SNODP is actual snow depth (m); ERA5 sd is snow water equivalent (m).
# Both use 0.01 m threshold. Any detectable snow cover suppresses dust emission.

# ===========================================================================
# Helper functions
# ===========================================================================

def air_density(t2m, d2m, sp):
    """Air density from temperature, dewpoint, and pressure [kg/m³]."""
    d2m_c = d2m - 273.15
    e_sat = 611.21 * np.exp((18.678 - d2m_c / 234.5) * (d2m_c / (257.14 + d2m_c)))
    R_d, R_v = 287.05, 461.5
    w = (R_d / R_v) * e_sat / (sp - e_sat)
    T_v = t2m * (1 + w / (R_d / R_v)) / (1 + w)
    return sp / (R_d * T_v)


def median_particle_diameter(clay_frac, silt_frac, lai):
    """
    Spatially varying median soil particle diameter [m].

    Leung 2023 Eq. (A1):
        Arid (LAI < 1):     D_p = 127 ± 47 µm (global constant)
        Non-arid (LAI ≥ 1): D_p = Psi_0 + Psi_1 × (f_silt + f_clay)
    """
    C = CONST
    D_p = np.full_like(clay_frac, C['D_p_arid'])

    # Non-arid: use regression on silt+clay fraction
    nonarid = lai >= C['LAI_thr']
    f_fine = (clay_frac[nonarid] + silt_frac[nonarid]) / 100.0  # fraction 0-1
    D_p[nonarid] = C['Psi_0'] + C['Psi_1'] * f_fine

    # Clamp to physically reasonable range
    D_p = np.clip(D_p, 10e-6, 500e-6)
    return D_p


# ===========================================================================
# Input data loader
# ===========================================================================

class InputData:
    """Loads meteorological driving data (ERA5 or MERRA-2), Prigent z0a, MODIS LAI,
    MODIS land cover, and SoilGrids data."""

    def __init__(self, meteo_path, soilgrids_dir, prigent_file, modis_lai_dir,
                 landcover_file=None, meteo_source='era5'):
        """meteo_path: ERA5 or MERRA-2 file/directory. meteo_source: 'era5' or 'merra2'."""
        self.meteo_source = meteo_source
        self.era5_path = self.meteo_path = Path(meteo_path)
        self.soilgrids_dir = Path(soilgrids_dir)
        self.prigent_file = prigent_file
        self.modis_lai_dir = Path(modis_lai_dir)
        self.landcover_file = landcover_file

        # Multi-file meteo support (ERA5 or MERRA-2)
        self._era5_files = []      # list of (filepath, n_times) tuples
        self._era5_file_times = [] # list of (filepath, start_global_idx, end_global_idx)
        self._era5_ds = None       # currently open dataset
        self._era5_current_file = None  # path of currently open file

        # Grid
        self.lon = None
        self.lat = None
        self.times = None
        self.nt = self.nlon = self.nlat = 0

        # Static fields
        self.lsm = None       # land-sea mask [0/1]
        self.z0a = None        # Prigent aeolian roughness [m]
        self.clay = None       # clay fraction [%]
        self.silt = None       # silt fraction [%]
        self.bulk_den = None   # bulk density [kg/m³]
        self.f_erodible = None # erodibility fraction from MODIS land cover [0-1]

        # MODIS LAI lookup
        self._lai_files = {}   # {date_ordinal: filepath}
        self._lai_dates = []   # sorted date ordinals
        self._lai_cache = {}   # {date_ordinal: (lat, lon) array on ERA5 grid}

    def load(self):
        """Load all input datasets."""
        if self.meteo_source == 'era5':
            self._load_era5_grid()
        elif self.meteo_source == 'merra2':
            self._load_merra2_grid()
        else:
            raise ValueError(f"Unknown meteo_source: {self.meteo_source!r} (use 'era5' or 'merra2')")
        self._load_prigent()
        self._load_soilgrids()
        self._load_landcover()
        self._index_modis_lai()
        log.info(f"Grid: {self.nlon}×{self.nlat}, {self.nt} timesteps")

    def _load_era5_grid(self):
        """Open ERA5 file(s) and read grid, static fields.

        Supports single file or directory of monthly files (era5_dust_YYYYMM.nc).
        Monthly files are opened sequentially to keep memory low.
        """
        if self.era5_path.is_dir():
            # Find all ERA5 NetCDF files in directory, sorted by name
            files = sorted(self.era5_path.glob('era5_dust_*.nc'))
            if not files:
                raise FileNotFoundError(
                    f"No era5_dust_*.nc files in {self.era5_path}")
            log.info(f"ERA5: {self.era5_path} ({len(files)} files)")
        else:
            files = [self.era5_path]
            log.info(f"ERA5: {self.era5_path}")

        # Read grid from first file
        ds0 = nc4.Dataset(str(files[0]), 'r')
        self.lon = np.array(ds0.variables['longitude'][:])
        self.lat = np.array(ds0.variables['latitude'][:])
        self.nlon = len(self.lon)
        self.nlat = len(self.lat)

        # Collect times from all files
        all_times = []
        global_idx = 0
        for f in files:
            ds = nc4.Dataset(str(f), 'r')
            time_name = 'valid_time' if 'valid_time' in ds.variables else 'time'
            time_var = ds.variables[time_name]
            cal = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
            ftimes = nc4.num2date(time_var[:], time_var.units, cal)
            n = len(ftimes)
            self._era5_file_times.append((str(f), global_idx, global_idx + n))
            all_times.extend(ftimes)
            global_idx += n
            if ds is not ds0:
                ds.close()

        self.times = np.array(all_times)
        self.nt = len(self.times)

        # Open first file as current
        self._era5_ds = ds0
        self._era5_current_file = str(files[0])

        # Land-sea mask (threshold at 0.5, remove snow in physics)
        lsm_raw = self._read_era5('lsm', static=True)
        self.lsm = np.where(lsm_raw >= 0.5, 1.0, 0.0)

        log.info(f"  Domain: lon [{self.lon.min():.1f}, {self.lon.max():.1f}], "
                 f"lat [{self.lat.min():.1f}, {self.lat.max():.1f}]")
        log.info(f"  Period: {self.times[0]} to {self.times[-1]}")
        if len(files) > 1:
            log.info(f"  Files: {len(files)}, total timesteps: {self.nt}")

    def _load_merra2_grid(self):
        """Open MERRA-2 monthly file(s) and read grid + static land mask.

        Expects files named merra2_dust_YYYYMM.nc with dimensions (time, lat, lon).
        MERRA-2 native grid: 0.5° lat × 0.625° lon (361 × 576).
        FRLAND (land fraction, static 2D) is used as land-sea mask.
        """
        if self.meteo_path.is_dir():
            files = sorted(self.meteo_path.glob('merra2_dust_*.nc'))
            if not files:
                raise FileNotFoundError(
                    f"No merra2_dust_*.nc files in {self.meteo_path}")
            log.info(f"MERRA-2: {self.meteo_path} ({len(files)} files)")
        else:
            files = [self.meteo_path]
            log.info(f"MERRA-2: {self.meteo_path}")

        ds0 = nc4.Dataset(str(files[0]), 'r')

        # MERRA-2 uses 'lat'/'lon' dimension names
        self.lon = np.array(ds0.variables['lon'][:])
        self.lat = np.array(ds0.variables['lat'][:])
        self.nlon = len(self.lon)
        self.nlat = len(self.lat)

        # Collect times from all files (reuse ERA5 infrastructure)
        all_times = []
        global_idx = 0
        for f in files:
            ds = nc4.Dataset(str(f), 'r')
            time_var = ds.variables['time']
            cal = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
            ftimes = nc4.num2date(time_var[:], time_var.units, cal)
            n = len(ftimes)
            self._era5_file_times.append((str(f), global_idx, global_idx + n))
            all_times.extend(ftimes)
            global_idx += n
            if ds is not ds0:
                ds.close()

        self.times = np.array(all_times)
        self.nt = len(self.times)

        self._era5_ds = ds0
        self._era5_current_file = str(files[0])

        # Land mask: prefer DISPH (displacement height) — land where DISPH is finite and >0.
        # Equivalent to Leung 2023 landfilt = DISPH/DISPH (1 over land, NaN over ocean).
        # Fallback: derive from SFMC validity (land cells have finite soil moisture values).
        if 'DISPH' in ds0.variables:
            disph_raw = np.array(ds0.variables['DISPH'][0])  # first timestep
            disph_raw = np.where(np.abs(disph_raw) > 1e10, np.nan, disph_raw)
            self.lsm = np.where(np.isfinite(disph_raw) & (disph_raw >= 0), 1.0, 0.0)
            log.info(f"  LSM from DISPH: {int(self.lsm.sum())} land cells")
        else:
            sfmc_var = MERRA2_VARS['swvl1']  # 'SFMC'
            if sfmc_var in ds0.variables:
                sfmc_raw = np.array(ds0.variables[sfmc_var][0])
                sfmc_raw = np.where(np.abs(sfmc_raw) > 1e10, np.nan, sfmc_raw)
                self.lsm = np.where(np.isfinite(sfmc_raw) & (sfmc_raw >= 0), 1.0, 0.0)
                log.info(f"  LSM derived from SFMC validity: {int(self.lsm.sum())} land cells")
            else:
                log.warning(f"  No DISPH or SFMC found — lsm = 1 everywhere")
                self.lsm = np.ones((self.nlat, self.nlon))

        log.info(f"  Domain: lon [{self.lon.min():.1f}, {self.lon.max():.1f}], "
                 f"lat [{self.lat.min():.1f}, {self.lat.max():.1f}]")
        log.info(f"  Period: {self.times[0]} to {self.times[-1]}")
        if len(files) > 1:
            log.info(f"  Files: {len(files)}, total timesteps: {self.nt}")

    def _ensure_era5_file(self, global_tidx):
        """Switch to the correct meteo file for the given global timestep index."""
        for fpath, start, end in self._era5_file_times:
            if start <= global_tidx < end:
                if fpath != self._era5_current_file:
                    if self._era5_ds:
                        self._era5_ds.close()
                    self._era5_ds = nc4.Dataset(fpath, 'r')
                    self._era5_current_file = fpath
                    log.info(f"  Switched to: {Path(fpath).name}")
                return global_tidx - start  # local index within file
        raise IndexError(f"Global timestep {global_tidx} not in any ERA5 file")

    def _read_era5(self, varname, static=False, tidx=None):
        """Read ERA5 field. Returns (lat, lon) array.

        tidx is local (within-file) index — use _ensure_era5_file() first.
        """
        ds = self._era5_ds
        if varname not in ds.variables:
            log.warning(f"  ERA5 '{varname}' missing — zeros")
            return np.zeros((self.nlat, self.nlon))
        var = ds.variables[varname]
        if static or var.ndim == 2:
            data = np.array(var[:])
            return data[0] if data.ndim == 3 else data
        elif tidx is not None:
            return np.array(var[tidx])
        return np.array(var[:])

    def get_timestep(self, tidx):
        """Read time-varying meteo fields for one timestep. Returns standard internal dict."""
        if self.meteo_source == 'era5':
            return self._get_era5_timestep(tidx)
        else:
            return self._get_merra2_timestep(tidx)

    def _get_era5_timestep(self, tidx):
        """Read ERA5 fields — returns {zust, swvl1, t2m, d2m, sp, sd, [blh], [u10, v10]}."""
        local_tidx = self._ensure_era5_file(tidx)
        fields = {}
        for v in ['zust', 'swvl1', 't2m', 'd2m', 'sp', 'sd']:
            fields[v] = self._read_era5(v, tidx=local_tidx)
        # BLH is optional (for intermittency)
        if 'blh' in self._era5_ds.variables:
            fields['blh'] = self._read_era5('blh', tidx=local_tidx)
        return fields

    def _get_merra2_timestep(self, tidx):
        """Read MERRA-2 fields — returns same internal keys as ERA5 path.

        MERRA-2 provides RHOA directly → 'rhoa' key instead of t2m/d2m/sp.
        compute_emission() detects 'rhoa' in met dict and skips air_density().
        Snow: SHLAND is actual depth (m); threshold 0.01 m ≈ any detectable cover.
        """
        local_tidx = self._ensure_era5_file(tidx)
        fields = {}
        # Read using MERRA-2 variable names, store as internal keys
        internal_to_m2 = {
            'zust':  MERRA2_VARS['zust'],   # USTAR → friction velocity
            'rhoa':  MERRA2_VARS['rhoa'],   # RHOA  → air density (skip air_density())
            'swvl1': MERRA2_VARS['swvl1'],  # SFMC  → surface soil moisture
            'sd':    MERRA2_VARS['sd'],     # SHLAND → snow depth (actual, not SWE)
        }
        for internal, m2var in internal_to_m2.items():
            fields[internal] = self._read_era5(m2var, tidx=local_tidx)
        # BLH is optional
        if MERRA2_VARS['blh'] in self._era5_ds.variables:
            fields['blh'] = self._read_era5(MERRA2_VARS['blh'], tidx=local_tidx)
        return fields

    def _load_prigent(self):
        """Load Prigent et al. (2005) aeolian roughness and regrid to ERA5."""
        log.info(f"Prigent z0a: {self.prigent_file}")
        with nc4.Dataset(self.prigent_file, 'r') as ds:
            p_lon = np.array(ds.variables['lon'][:])
            p_lat = np.array(ds.variables['lat'][:])
            z0a_cm = np.array(ds.variables['Z0a'][0, :, :])

        # Convert cm → m
        z0a_m = z0a_cm / 100.0
        z0a_m[z0a_m <= 0] = np.nan

        # Regrid to ERA5 grid
        self.z0a = self._regrid(p_lon, p_lat, z0a_m, fill=np.nan)

        valid = self.z0a[np.isfinite(self.z0a) & (self.z0a > 0)]
        log.info(f"  z0a: {len(valid)} valid cells, "
                 f"median={np.nanmedian(valid):.2e} m")

    def _load_soilgrids(self):
        """Load SoilGrids clay, silt, bulk density."""
        sg = self.soilgrids_dir
        log.info(f"SoilGrids: {sg}")

        self.clay = self._load_soil_file(sg / 'SG_clay.nc4', 'T_CLAY',
                                          sg / 'T_CLAY.nc4', 'clay')
        self.silt = self._load_soil_file(sg / 'SG_silt.nc4', 'T_SILT',
                                          sg / 'T_SILT.nc4', 'silt')

        # Bulk density
        bd_file = sg / 'SG_bdod.nc4'
        if bd_file.exists():
            self.bulk_den = self._load_soil_nc(str(bd_file), 'bdod', 'bulk density')
        else:
            log.warning("  Bulk density not found — estimating")
            porosity = 0.45 + 0.05 * (self.clay / 100.0)
            self.bulk_den = (1.0 - porosity) * CONST['rho_p']

        # Clean
        self.clay = np.clip(np.nan_to_num(self.clay, nan=10.0), 0, 100)
        self.silt = np.clip(np.nan_to_num(self.silt, nan=20.0), 0, 100)
        self.bulk_den[self.bulk_den <= 0] = 1500.0

        log.info(f"  Clay: mean={np.nanmean(self.clay):.1f}%")
        log.info(f"  Silt: mean={np.nanmean(self.silt):.1f}%")

    def _load_soil_file(self, primary, primary_var, fallback, desc):
        """Try primary SoilGrids file, fallback to HWSD."""
        if primary.exists():
            return self._load_soil_nc(str(primary), primary_var, desc)
        elif fallback.exists():
            log.info(f"  Fallback for {desc}: {fallback}")
            return self._load_soil_nc(str(fallback), primary_var, desc)
        else:
            log.warning(f"  {desc} not found — default 10%")
            return np.full((self.nlat, self.nlon), 10.0)

    def _load_soil_nc(self, filepath, varname, desc):
        """Load a 2D soil NetCDF and regrid to ERA5 grid."""
        log.info(f"  Loading {desc}: {filepath}")
        with nc4.Dataset(filepath, 'r') as ds:
            data = np.array(ds.variables[varname][:])
            if data.ndim == 3:
                data = data[0]
            src_lon = src_lat = None
            for ln in ['lon', 'longitude']:
                if ln in ds.variables:
                    src_lon = np.array(ds.variables[ln][:])
                    break
            for lt in ['lat', 'latitude']:
                if lt in ds.variables:
                    src_lat = np.array(ds.variables[lt][:])
                    break

        if src_lon is None or src_lat is None:
            return data

        if len(src_lon) == self.nlon and len(src_lat) == self.nlat:
            return data

        return self._regrid(src_lon, src_lat, data, fill=np.nan)

    def _regrid(self, src_lon, src_lat, data, fill=0.0):
        """Nearest-neighbor regrid to ERA5 grid."""
        from scipy.interpolate import RegularGridInterpolator

        # Ensure lat is monotonically increasing for interpolator
        if src_lat[0] > src_lat[-1]:
            src_lat = src_lat[::-1]
            data = data[::-1, :]

        data = data.astype(np.float64)
        data[data < -900] = np.nan

        interp = RegularGridInterpolator(
            (src_lat, src_lon), data,
            method='nearest', bounds_error=False, fill_value=fill
        )

        # ERA5 lat may be N→S, interpolator needs actual values
        lat_grid, lon_grid = np.meshgrid(self.lat, self.lon, indexing='ij')
        pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        return interp(pts).reshape(self.nlat, self.nlon)

    # --- MODIS Land Cover ---

    def _load_landcover(self):
        """Load MODIS MCD12C1 erodibility mask and regrid to ERA5."""
        if self.landcover_file is None:
            log.warning("No land cover file — f_erodible = 1.0 everywhere "
                        "(all land cells can emit)")
            self.f_erodible = self.lsm.copy()
            return

        log.info(f"Land cover: {self.landcover_file}")
        with nc4.Dataset(self.landcover_file, 'r') as ds:
            lc_lon = np.array(ds.variables['lon'][:])
            lc_lat = np.array(ds.variables['lat'][:])
            # Use f_erodible (weighted barren+shrubland+savanna+grassland)
            f_erod = np.array(ds.variables['f_erodible'][:])

        # Regrid 0.05° → ERA5 grid
        self.f_erodible = self._regrid(lc_lon, lc_lat, f_erod, fill=0.0)
        self.f_erodible = np.clip(self.f_erodible, 0.0, 1.0)

        erod_cells = np.sum(self.f_erodible > 0)
        land_cells = np.sum(self.lsm > 0.5)
        log.info(f"  Erodible cells: {erod_cells}/{land_cells} land cells "
                 f"({100*erod_cells/max(land_cells,1):.1f}%)")
        log.info(f"  f_erodible mean (where>0): "
                 f"{self.f_erodible[self.f_erodible>0].mean():.3f}")

    # --- MODIS LAI ---

    def _index_modis_lai(self):
        """Index available MODIS LAI files by date."""
        pattern = str(self.modis_lai_dir / '**' / '*.nc')
        files = sorted(glob(pattern, recursive=True))
        if not files:
            log.warning(f"  No MODIS LAI files found in {self.modis_lai_dir}")
            return

        for f in files:
            # Extract date from filename: ...UHAM-ICDC__YYYYMMDD__fv0.03.nc
            basename = os.path.basename(f)
            parts = basename.split('__')
            for p in parts:
                if len(p) == 8 and p.isdigit():
                    dt = datetime(int(p[:4]), int(p[4:6]), int(p[6:8]))
                    ordinal = dt.toordinal()
                    self._lai_files[ordinal] = f
                    break

        self._lai_dates = sorted(self._lai_files.keys())
        log.info(f"MODIS LAI: {len(self._lai_dates)} composites indexed")

    def get_modis_lai(self, dt):
        """
        Get MODIS LAI for a given datetime, interpolated to ERA5 grid.

        Uses nearest 8-day composite. Caches regridded arrays.
        """
        if not self._lai_dates:
            return np.zeros((self.nlat, self.nlon))

        # Find nearest composite date
        target = dt.toordinal() if hasattr(dt, 'toordinal') else \
                 datetime(dt.year, dt.month, dt.day).toordinal()
        idx = np.searchsorted(self._lai_dates, target)
        idx = min(idx, len(self._lai_dates) - 1)
        # Check if previous date is closer
        if idx > 0:
            if abs(self._lai_dates[idx - 1] - target) < abs(self._lai_dates[idx] - target):
                idx -= 1
        nearest = self._lai_dates[idx]

        # Return cached if available
        if nearest in self._lai_cache:
            return self._lai_cache[nearest]

        # Load and regrid
        f = self._lai_files[nearest]
        with nc4.Dataset(f, 'r') as ds:
            lai_lon = np.array(ds.variables['lon'][:])
            lai_lat = np.array(ds.variables['lat'][:])
            lai_var = ds.variables['lai']
            lai_data = np.array(lai_var[0, :, :])

            # Get fill value from attributes
            fill_val = getattr(lai_var, '_FillValue', 255.0)
            if hasattr(lai_var, 'missing_value'):
                fill_val = lai_var.missing_value

        # Handle fill values (255 = no retrieval in MODIS LAI)
        lai_data[lai_data >= fill_val] = 0.0
        if hasattr(lai_data, 'filled'):
            lai_data = lai_data.filled(0.0)
        lai_data = np.nan_to_num(lai_data, nan=0.0)
        lai_data = np.clip(lai_data, 0.0, 10.0)

        # Regrid 0.5° → ERA5 grid
        if len(lai_lon) == self.nlon and len(lai_lat) == self.nlat:
            lai_regridded = lai_data
        else:
            lai_regridded = self._regrid(lai_lon, lai_lat, lai_data, fill=0.0)

        self._lai_cache[nearest] = lai_regridded
        return lai_regridded

    def close(self):
        if self._era5_ds:
            self._era5_ds.close()
            self._era5_ds = None


# ===========================================================================
# Dust emission physics — Leung et al. (2023)
# ===========================================================================

def compute_emission(met, static, lai, C_tune=0.05):
    """
    Compute dust emission flux for a single timestep — Leung et al. (2023).

    Implements Kok et al. (2014) emission with MB95+Okin drag partition.

    Parameters
    ----------
    met : dict — ERA5 fields {zust, swvl1, t2m, d2m, sp, sd, [blh]}
    static : dict — {lsm, z0a, clay, silt, bulk_den, D_p, z0s, f_erodible}
    lai : ndarray (lat, lon) — MODIS LAI [m²/m²]
    C_tune : float — global tuning constant

    Returns
    -------
    F_d : ndarray (lat, lon) — dust emission flux [kg/m²/s]
    diag : dict — diagnostic fields
    """
    C = CONST

    # ------------------------------------------------------------------
    # 1. Masking: land + no snow
    # ------------------------------------------------------------------
    mask = static['lsm'].copy()
    mask[met['sd'] > 0.01] = 0.0  # snow-covered → no emission

    # ------------------------------------------------------------------
    # 2. Air density [kg/m³]
    #    ERA5: computed from t2m, d2m, sp via Magnus formula
    #    MERRA-2: provided directly as RHOA — use as-is
    # ------------------------------------------------------------------
    if 'rhoa' in met:
        rhoa = met['rhoa']          # MERRA-2 path
    else:
        rhoa = air_density(met['t2m'], met['d2m'], met['sp'])  # ERA5 path

    # ------------------------------------------------------------------
    # 3. Median soil particle diameter D_p [m]
    #    Spatially varying: arid (LAI<1) = 127 µm, else from clay+silt
    # ------------------------------------------------------------------
    D_p = static['D_p']
    z0s = static['z0s']

    # ------------------------------------------------------------------
    # 4. Dry fluid threshold u*_ft0 — Shao & Lu (2000)
    # ------------------------------------------------------------------
    ustar_ft0 = np.sqrt(
        C['A_SL'] * (C['rho_p'] * C['g'] * D_p + C['gamma_SL'] / D_p)
    ) / np.sqrt(rhoa)

    # ------------------------------------------------------------------
    # 5. Impact threshold u*_it = B_it × u*_ft0
    #    Used as emission threshold (not wet fluid threshold)
    # ------------------------------------------------------------------
    ustar_it = C['B_it'] * ustar_ft0

    # ------------------------------------------------------------------
    # 6. Soil moisture correction — Fécan et al. (1999)
    #    Only affects fluid threshold (for C_d and κ calculations)
    # ------------------------------------------------------------------
    swvl1 = np.clip(met['swvl1'], 0.0, 0.7)
    gwc = C['rho_water'] * swvl1 / static['bulk_den']  # gravimetric

    clay_pct = static['clay']
    gwc_thr = C['a_moist'] * (0.17 * clay_pct + 0.0014 * clay_pct ** 2) / 100.0

    f_moisture = np.ones_like(gwc)
    wet = gwc > gwc_thr
    f_moisture[wet] = np.sqrt(1.0 + 1.21 * (100.0 * (gwc[wet] - gwc_thr[wet])) ** 0.68)

    ustar_ft_wet = ustar_ft0 * f_moisture

    # ------------------------------------------------------------------
    # 7. Rock drag partition — Marticorena & Bergametti (1995)
    #    Using Prigent et al. (2005) satellite aeolian roughness
    # ------------------------------------------------------------------
    z0a = static['z0a'].copy()
    z0a_valid = np.isfinite(z0a) & (z0a > 0)

    f_eff_r = np.ones(rhoa.shape)
    # MB95: f_eff = 1 - ln(z0a/z0s) / ln(0.7 × (X/z0s)^0.8)
    denom = np.log(C['b1'] * (C['X'] / z0s[z0a_valid]) ** C['b2'])
    numer = np.log(z0a[z0a_valid] / z0s[z0a_valid])
    f_eff_r[z0a_valid] = 1.0 - numer / denom
    f_eff_r = np.clip(f_eff_r, 0.001, 1.0)

    # ------------------------------------------------------------------
    # 8. Vegetation drag partition — Okin (2008)
    #    f_eff,v = f0 + (1-f0) × K/(K+c)
    #    K = π / (2 × VAI) ≈ mean gap / vegetation height
    # ------------------------------------------------------------------
    vai = np.clip(lai, 0.0, None)  # VAI ≈ LAI (no SAI data)
    K_okin = np.where(vai > 0.001, np.pi / (2.0 * np.maximum(vai, 1e-6)), 1e6)  # large K = sparse

    f_eff_v = C['f0'] + (1.0 - C['f0']) * K_okin / (K_okin + C['c_okin'])
    f_eff_v = np.clip(f_eff_v, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 9. Combined drag partition → saltation friction velocity
    #    u*_s = u* × f_eff_r × f_eff_v
    # ------------------------------------------------------------------
    F_eff = f_eff_r * f_eff_v
    wnd_frc_slt = met['zust'] * F_eff

    # ------------------------------------------------------------------
    # 10. Bare soil fraction
    #     f_bare = f_erodible × max(0, 1 - LAI/LAI_thr) × land_mask
    #     f_erodible: from MODIS MCD12C1 land cover (barren + shrubland)
    #     LAI modulation: vegetation growth reduces emission within
    #     erodible cells (e.g., Sahel seasonal greening)
    # ------------------------------------------------------------------
    f_lai = np.clip(1.0 - lai / C['LAI_thr'], 0.0, 1.0)
    f_bare = static['f_erodible'] * f_lai * mask

    # ------------------------------------------------------------------
    # 11. Standardized fluid threshold
    #     u*_st = u*_ft_wet × sqrt(ρ_a / ρ_a0)
    # ------------------------------------------------------------------
    ustar_st = ustar_ft_wet * np.sqrt(rhoa / C['rho_a0'])

    # ------------------------------------------------------------------
    # 12. Soil erodibility coefficient C_d
    #     C_d = C_d0 × exp(-C_e × (u*_st - u*_st0) / u*_st0)
    # ------------------------------------------------------------------
    C_d = C['C_d0'] * np.exp(-C['C_e'] * (ustar_st - C['ustar_st0']) / C['ustar_st0'])

    # ------------------------------------------------------------------
    # 13. Fragmentation exponent κ (capped at 3)
    #     κ = C_α × (u*_st - u*_st0) / u*_st0
    # ------------------------------------------------------------------
    kappa = C['C_alpha'] * (ustar_st - C['ustar_st0']) / C['ustar_st0']
    kappa = np.clip(kappa, 0.0, C['kappa_cap'])

    # ------------------------------------------------------------------
    # 14. Clay mass fraction for emission scaling
    #     Kok 2014: f_clay is mass fraction (0-1), capped at 0.20
    # ------------------------------------------------------------------
    clay_scale = np.clip(clay_pct / 100.0, 0.0, 0.20)

    # ------------------------------------------------------------------
    # 15. Intermittency factor η — Comola et al. (2019)
    #     Fraction of time when saltation is active within model timestep
    # ------------------------------------------------------------------
    eta = np.ones_like(rhoa)
    if 'blh' in met and np.any(met['blh'] > 0):
        from scipy.special import erf as sp_erf

        # Wind standard deviation from MOST (neutral approx)
        # σ_u ≈ 2.3 × u* (Panofsky et al. 1977)
        sigma_u = 2.3 * met['zust']
        sigma_u = np.clip(sigma_u, 0.01, None)

        # Convert thresholds to wind speed at 10m
        z_ref = 10.0  # reference height
        z0a_arr = static['z0a']
        z0_ref = np.where(z0a_valid, z0a_arr, 3e-4)
        log_ratio = np.log(z_ref / z0_ref) / C['k_vk']

        u_mean = met['zust'] * log_ratio
        u_ft = ustar_ft_wet * log_ratio
        u_it = ustar_it * log_ratio

        sqrt2 = np.sqrt(2.0)
        # P(u > u_ft): probability of exceeding fluid threshold
        p_above_ft = 0.5 * (1.0 - sp_erf((u_ft - u_mean) / (sqrt2 * sigma_u)))
        # P(u > u_it): probability of exceeding impact threshold
        p_above_it = 0.5 * (1.0 - sp_erf((u_it - u_mean) / (sqrt2 * sigma_u)))

        # Threshold crossing rate
        with np.errstate(over='ignore', invalid='ignore'):
            exp_arg = (u_ft ** 2 - u_it ** 2
                       - 2.0 * u_mean * (u_ft - u_it)) / (2.0 * sigma_u ** 2)
            cross_rate = 1.0 / (np.exp(exp_arg) + 1.0)

        # Intermittency: fraction of time with active saltation
        eta = p_above_ft + cross_rate * (p_above_it - p_above_ft)
        eta = np.clip(np.nan_to_num(eta, nan=0.0), 0.0, 1.0)

    # ------------------------------------------------------------------
    # 16. Dust emission flux — Kok et al. (2014) Eq. (1)
    #     F_d = C_tune × C_d × f_bare × f_clay × ρ_a ×
    #           (u*_s² - u*_it²) / u*_it × (u*_s / u*_it)^κ × η
    # ------------------------------------------------------------------
    F_d = np.zeros_like(rhoa)
    active = (wnd_frc_slt > ustar_it) & (mask > 0.5) & (f_bare > 0)

    if np.any(active):
        u_s = wnd_frc_slt[active]
        u_t = ustar_it[active]

        F_d[active] = (
            C_tune
            * C_d[active]
            * f_bare[active]
            * clay_scale[active]
            * rhoa[active]
            * (u_s ** 2 - u_t ** 2) / u_t
            * (u_s / u_t) ** kappa[active]
            * eta[active]
        )

    # Clean
    F_d = np.nan_to_num(F_d, nan=0.0, posinf=0.0, neginf=0.0)
    F_d = np.clip(F_d, 0.0, None)

    diag = {
        'rhoa': rhoa, 'ustar_ft0': ustar_ft0, 'ustar_ft_wet': ustar_ft_wet,
        'ustar_it': ustar_it, 'wnd_frc_slt': wnd_frc_slt,
        'f_bare': f_bare, 'f_eff_r': f_eff_r, 'f_eff_v': f_eff_v,
        'F_eff': F_eff, 'kappa': kappa, 'C_d': C_d, 'eta': eta,
        'D_p': D_p, 'f_moisture': f_moisture,
    }
    return F_d, diag


# ===========================================================================
# Output writer
# ===========================================================================

def _write_monthly(output_dir, ym, buf, lon, lat, all_times):
    """Write one month of timestep data to NetCDF."""
    year, month = ym
    filepath = os.path.join(output_dir, f'odem_{year}{month:02d}.nc')
    log.info(f"  Writing monthly: {filepath} ({len(buf)} timesteps)")

    times = [all_times[tidx] for tidx, _ in buf]
    F_d_arr = np.stack([fd for _, fd in buf], axis=0)

    with nc4.Dataset(filepath, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(times))
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))

        t_v = ds.createVariable('time', 'f8', ('time',))
        t_v.units = f'hours since {times[0].strftime("%Y-%m-%d %H:%M:%S")}'
        t_v.calendar = 'standard'
        t_v[:] = nc4.date2num(times, t_v.units, t_v.calendar)

        lat_v = ds.createVariable('lat', 'f8', ('lat',))
        lat_v[:] = lat
        lat_v.units = 'degrees_north'

        lon_v = ds.createVariable('lon', 'f8', ('lon',))
        lon_v[:] = lon
        lon_v.units = 'degrees_east'

        fd_v = ds.createVariable('F_d', 'f4', ('time', 'lat', 'lon'),
                                  zlib=True, complevel=4)
        fd_v[:] = F_d_arr
        fd_v.units = 'kg m-2 s-1'
        fd_v.long_name = 'Dust emission flux (Leung et al. 2023)'

        ds.title = f'ODEM — Offline Dust Emission Model — {year}-{month:02d}'
        ds.source = 'Kok et al. (2014) + Leung et al. (2023)'
        ds.history = f'Created {datetime.now().isoformat()}'


def _write_annual_mean(filepath, lon, lat, F_mean, F_max, t_start, t_end, nt):
    """Write annual mean emission field to NetCDF."""
    log.info(f"Writing annual mean: {filepath}")
    with nc4.Dataset(filepath, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))

        lat_v = ds.createVariable('lat', 'f8', ('lat',))
        lat_v[:] = lat
        lat_v.units = 'degrees_north'

        lon_v = ds.createVariable('lon', 'f8', ('lon',))
        lon_v[:] = lon
        lon_v.units = 'degrees_east'

        fm_v = ds.createVariable('F_d_mean', 'f4', ('lat', 'lon'),
                                  zlib=True, complevel=4)
        fm_v[:] = F_mean
        fm_v.units = 'kg m-2 s-1'
        fm_v.long_name = 'Time-mean dust emission flux'

        fx_v = ds.createVariable('F_d_max', 'f4', ('lat', 'lon'),
                                  zlib=True, complevel=4)
        fx_v[:] = F_max
        fx_v.units = 'kg m-2 s-1'
        fx_v.long_name = 'Maximum dust emission flux'

        ds.title = 'ODEM — Offline Dust Emission Model — annual mean'
        ds.source = 'Kok et al. (2014) + Leung et al. (2023)'
        ds.references = ('Leung et al. (2023) ACP 23:6487-6523; '
                         'Kok et al. (2014) ACP 14:13023-13041')
        ds.history = f'Created {datetime.now().isoformat()}'
        ds.institution = 'Standalone model (Baykara)'
        ds.period = f'{t_start} to {t_end}'
        ds.n_timesteps = nt


def write_output(filepath, lon, lat, times, F_d_all):
    """Write emission output to NetCDF (legacy — for small runs)."""
    log.info(f"Writing: {filepath}")
    with nc4.Dataset(filepath, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(times))
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))

        t_v = ds.createVariable('time', 'f8', ('time',))
        t_v.units = f'hours since {times[0].strftime("%Y-%m-%d %H:%M:%S")}'
        t_v.calendar = 'standard'
        t_v[:] = nc4.date2num(times, t_v.units, t_v.calendar)

        lat_v = ds.createVariable('lat', 'f8', ('lat',))
        lat_v[:] = lat
        lat_v.units = 'degrees_north'

        lon_v = ds.createVariable('lon', 'f8', ('lon',))
        lon_v[:] = lon
        lon_v.units = 'degrees_east'

        fd_v = ds.createVariable('F_d', 'f4', ('time', 'lat', 'lon'),
                                  zlib=True, complevel=4)
        fd_v[:] = F_d_all
        fd_v.units = 'kg m-2 s-1'
        fd_v.long_name = 'Dust emission flux (Leung et al. 2023)'

        ds.title = 'ODEM — Offline Dust Emission Model'
        ds.source = 'Kok et al. (2014) + Leung et al. (2023)'
        ds.references = ('Leung et al. (2023) ACP 23:6487-6523; '
                         'Kok et al. (2014) ACP 14:13023-13041')
        ds.history = f'Created {datetime.now().isoformat()}'
        ds.institution = 'Standalone model (Baykara)'


# ===========================================================================
# Main runner
# ===========================================================================

def _write_diag_mean(filepath, lon, lat, diag_sums, static, nt, t_start, t_end):
    """Write time-mean diagnostic fields to NetCDF for verification."""
    log.info(f"Writing diagnostics: {filepath}")

    DIAG_META = {
        'rhoa':         ('Air density',                    'kg m-3'),
        'ustar_ft0':    ('Dry fluid threshold u*_ft0',     'm s-1'),
        'ustar_ft_wet': ('Wet fluid threshold u*_ft_wet',  'm s-1'),
        'ustar_it':     ('Impact threshold u*_it',         'm s-1'),
        'wnd_frc_slt':  ('Saltation friction velocity',    'm s-1'),
        'f_bare':       ('Bare erodible soil fraction',    '1'),
        'f_eff_r':      ('Rock drag partition factor',     '1'),
        'f_eff_v':      ('Vegetation drag partition factor','1'),
        'F_eff':        ('Combined drag partition factor', '1'),
        'kappa':        ('Fragmentation exponent kappa',   '1'),
        'C_d':          ('Erodibility coefficient C_d',    '1'),
        'eta':          ('Intermittency factor eta',       '1'),
        'f_moisture':   ('Fécan soil moisture correction', '1'),
    }
    STATIC_META = {
        'D_p':       ('Median soil particle diameter',     'm'),
        'z0a':       ('Prigent aeolian roughness z0a',     'm'),
        'clay':      ('Clay fraction',                     'percent'),
        'silt':      ('Silt fraction',                     'percent'),
        'bulk_den':  ('Soil bulk density',                 'kg m-3'),
        'f_erodible':('MODIS erodibility fraction',        '1'),
        'lsm':       ('Land-sea mask',                     '1'),
    }

    with nc4.Dataset(filepath, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))

        lat_v = ds.createVariable('lat', 'f8', ('lat',))
        lat_v[:] = lat; lat_v.units = 'degrees_north'
        lon_v = ds.createVariable('lon', 'f8', ('lon',))
        lon_v[:] = lon; lon_v.units = 'degrees_east'

        for name, arr_sum in diag_sums.items():
            lname, units = DIAG_META[name]
            v = ds.createVariable(name, 'f4', ('lat', 'lon'), zlib=True, complevel=4)
            v[:] = arr_sum / nt
            v.long_name = lname; v.units = units

        for name, arr in static.items():
            if name not in STATIC_META:
                continue
            lname, units = STATIC_META[name]
            data = np.array(arr)
            v = ds.createVariable(name, 'f4', ('lat', 'lon'), zlib=True, complevel=4)
            v[:] = np.nan_to_num(data, nan=-9999.0)
            v.long_name = lname; v.units = units
            v.missing_value = np.float32(-9999.0)

        ds.title = 'ODEM — Time-mean diagnostic fields'
        ds.source = 'Kok et al. (2014) + Leung et al. (2023)'
        ds.period = f'{t_start} to {t_end}'
        ds.n_timesteps = nt
        ds.history = f'Created {datetime.now().isoformat()}'


def run_model(meteo_path, soilgrids_dir, prigent_file, modis_lai_dir,
              landcover_file=None, output_dir='output_v2', C_tune=0.05,
              save_diags=False, meteo_source='era5'):
    """Run ODEM — Offline Dust Emission Model.

    meteo_path:      ERA5 or MERRA-2 file/directory.
    meteo_source:    'era5' (default) or 'merra2'
    save_diags:      if True, write odem_diag_mean.nc with time-mean diagnostic fields.
    """
    t0 = time.time()

    log.info("=" * 60)
    log.info("  ODEM — Offline Dust Emission Model")
    log.info("=" * 60)
    log.info(f"  Meteo:      {meteo_path} [{meteo_source.upper()}]")
    log.info(f"  Prigent:    {prigent_file}")
    log.info(f"  MODIS LAI:  {modis_lai_dir}")
    log.info(f"  Land cover: {landcover_file or 'NONE (all land emits)'}")
    log.info(f"  SoilGrids:  {soilgrids_dir}")
    log.info(f"  C_tune:     {C_tune}")
    log.info("=" * 60)

    # Load inputs
    inputs = InputData(meteo_path, soilgrids_dir, prigent_file, modis_lai_dir,
                       landcover_file=landcover_file, meteo_source=meteo_source)
    inputs.load()

    # Precompute static fields
    # Use first MODIS LAI for D_p classification (arid vs non-arid)
    lai_first = inputs.get_modis_lai(inputs.times[0])
    D_p = median_particle_diameter(inputs.clay, inputs.silt, lai_first)
    z0s = D_p / 15.0  # smooth roughness length (White 2006)

    static = {
        'lsm': inputs.lsm,
        'z0a': inputs.z0a,
        'clay': inputs.clay,
        'silt': inputs.silt,
        'bulk_den': inputs.bulk_den,
        'D_p': D_p,
        'z0s': z0s,
        'f_erodible': inputs.f_erodible,
    }

    log.info(f"  D_p: median={np.nanmedian(D_p)*1e6:.0f} µm, "
             f"range=[{np.nanmin(D_p)*1e6:.0f}, {np.nanmax(D_p)*1e6:.0f}] µm")

    # Run timesteps — accumulate mean rather than storing all timesteps
    os.makedirs(output_dir, exist_ok=True)
    F_d_sum = np.zeros((inputs.nlat, inputs.nlon), dtype=np.float64)
    F_d_max = np.zeros((inputs.nlat, inputs.nlon), dtype=np.float32)

    # Diagnostic accumulators (only allocated if save_diags=True)
    DIAG_KEYS = ['rhoa', 'ustar_ft0', 'ustar_ft_wet', 'ustar_it', 'wnd_frc_slt',
                 'f_bare', 'f_eff_r', 'f_eff_v', 'F_eff', 'kappa', 'C_d', 'eta',
                 'f_moisture']
    diag_sums = {k: np.zeros((inputs.nlat, inputs.nlon), dtype=np.float64)
                 for k in DIAG_KEYS} if save_diags else {}

    # For monthly output files: group timesteps by month
    monthly_bufs = {}  # {(year,month): [list of F_d arrays]}

    for tidx in range(inputs.nt):
        t = inputs.times[tidx]
        t_str = str(t)
        if tidx % 8 == 0:
            log.info(f"  [{tidx+1}/{inputs.nt}] {t_str}")

        met = inputs.get_timestep(tidx)
        lai = inputs.get_modis_lai(t)
        F_d, diag = compute_emission(met, static, lai, C_tune)

        F_d_sum += F_d
        F_d_max = np.maximum(F_d_max, F_d)

        if save_diags:
            for k in DIAG_KEYS:
                diag_sums[k] += diag[k]

        # Buffer for monthly output
        ym = (t.year, t.month)
        if ym not in monthly_bufs:
            monthly_bufs[ym] = []
        monthly_bufs[ym].append((tidx, F_d))

        if tidx == 0:
            active = F_d > 0
            n_active = np.sum(active)
            log.info(f"    Active cells: {n_active}/{F_d.size} "
                     f"({100*n_active/F_d.size:.1f}%)")
            if n_active > 0:
                log.info(f"    F_d: mean={F_d[active].mean():.2e}, "
                         f"max={F_d.max():.2e} kg/m²/s")
                log.info(f"    f_eff_r: mean={diag['f_eff_r'][active].mean():.3f}")
                log.info(f"    f_eff_v: mean={diag['f_eff_v'][active].mean():.3f}")
                log.info(f"    eta: mean={diag['eta'][active].mean():.3f}")

        # Write completed months to disk to free memory
        completed = [k for k in monthly_bufs
                     if k != ym and len(monthly_bufs[k]) > 0]
        for cm in completed:
            _write_monthly(output_dir, cm, monthly_bufs[cm],
                           inputs.lon, inputs.lat, inputs.times)
            del monthly_bufs[cm]

    # Write any remaining month
    for cm in list(monthly_bufs.keys()):
        if monthly_bufs[cm]:
            _write_monthly(output_dir, cm, monthly_bufs[cm],
                           inputs.lon, inputs.lat, inputs.times)
            del monthly_bufs[cm]

    # Annual mean output — only written for multi-month runs.
    # Single-month runs would overwrite a valid annual mean with one month's data.
    n_months = len(set((t.year, t.month) for t in inputs.times))
    if n_months > 1:
        F_mean = F_d_sum / inputs.nt  # kg/m²/s
        out_file = os.path.join(output_dir, 'odem_annual_mean.nc')
        _write_annual_mean(out_file, inputs.lon, inputs.lat, F_mean, F_d_max,
                           inputs.times[0], inputs.times[-1], inputs.nt)
    else:
        log.info("  Skipping annual mean (single-month run)")

    # Diagnostic output
    if save_diags:
        diag_file = os.path.join(output_dir, 'odem_diag_mean.nc')
        _write_diag_mean(diag_file, inputs.lon, inputs.lat, diag_sums,
                         static, inputs.nt, inputs.times[0], inputs.times[-1])

    # Summary
    elapsed = time.time() - t0
    dlat = abs(inputs.lat[1] - inputs.lat[0]) if len(inputs.lat) > 1 else 0.25
    dlon = abs(inputs.lon[1] - inputs.lon[0]) if len(inputs.lon) > 1 else 0.25
    lat_rad = np.deg2rad(inputs.lat)
    cell_area = (dlat * np.pi / 180) * (dlon * np.pi / 180) * \
                6.371e6 ** 2 * np.cos(lat_rad)  # m²
    cell_area_2d = cell_area[:, np.newaxis] * np.ones((1, inputs.nlon))

    total_flux = np.sum(F_mean * cell_area_2d)  # kg/s
    annual_Tg = total_flux * 365.25 * 24 * 3600 / 1e9

    mean_active = np.mean(F_mean[F_mean > 0]) if np.any(F_mean > 0) else 0

    log.info("=" * 60)
    log.info(f"  Done in {elapsed:.1f}s — {output_dir}/")
    log.info(f"  Annual total: {annual_Tg:.0f} Tg/yr")
    log.info(f"  Reference: Kok 2021 PM20 5000 Tg/yr (C_tune=0.05 overshoots by ~2.5×, per Leung 2023)")
    log.info(f"  Mean F_d (active): {mean_active:.2e} kg/m²/s")
    log.info(f"  Max F_d: {np.max(F_d_max):.2e} kg/m²/s")
    log.info("=" * 60)

    inputs.close()
    return F_mean


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='ODEM — Offline Dust Emission Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ERA5 — single month
  python odem.py \\
      --era5 era5_data/era5_dust_200601.nc \\
      --soilgrids soilgrids_data \\
      --prigent prigent_data/Prigent_2005_roughness_0.25x0.25.nc \\
      --modis-lai modis_lai/2006 \\
      --landcover modis_landcover/modis_landcover_erodibility_2006.nc

  # MERRA-2 — single month
  python odem.py \\
      --merra2 merra2_data/merra2_dust_200601.nc \\
      --soilgrids soilgrids_data \\
      --prigent prigent_data/Prigent_2005_roughness_0.25x0.25.nc \\
      --modis-lai modis_lai/2006 \\
      --landcover modis_landcover/modis_landcover_erodibility_2006.nc
""")
    meteo_group = p.add_mutually_exclusive_group(required=True)
    meteo_group.add_argument('--era5',
                   help='ERA5 NetCDF file or directory (era5_dust_YYYYMM.nc, 0.25°)')
    meteo_group.add_argument('--merra2',
                   help='MERRA-2 NetCDF file or directory (merra2_dust_YYYYMM.nc, 0.5°×0.625°)')
    p.add_argument('--soilgrids', required=True,
                   help='SoilGrids directory (SG_clay.nc4, SG_silt.nc4, SG_bdod.nc4)')
    p.add_argument('--prigent', required=True,
                   help='Prigent z0a aeolian roughness NetCDF')
    p.add_argument('--modis-lai', required=True,
                   help='MODIS LAI directory (contains YYYYMMDD .nc files)')
    p.add_argument('--landcover', default=None,
                   help='MODIS MCD12C1 erodibility NetCDF (from download_modis_landcover.py)')
    p.add_argument('--output-dir', default='output_v2')
    p.add_argument('--C-tune', type=float, default=0.05)
    p.add_argument('--save-diags', action='store_true',
                   help='Save time-mean diagnostic fields to odem_diag_mean.nc '
                        '(recommended for verification runs)')

    args = p.parse_args()

    if args.era5:
        meteo_path = args.era5
        meteo_source = 'era5'
    else:
        meteo_path = args.merra2
        meteo_source = 'merra2'

    run_model(
        meteo_path=meteo_path,
        soilgrids_dir=args.soilgrids,
        prigent_file=args.prigent,
        modis_lai_dir=args.modis_lai,
        landcover_file=args.landcover,
        output_dir=args.output_dir,
        C_tune=args.C_tune,
        save_diags=args.save_diags,
        meteo_source=meteo_source,
    )


if __name__ == '__main__':
    main()
