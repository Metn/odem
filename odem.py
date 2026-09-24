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
  - v2 schemes: albedo-based drag partition (aem) and the DEAD scheme of
    Zender et al. (2003a) as configured in GEOS-Chem (dead_g, dead_z)

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
    'u10':   'U10M',     # 10 m eastward wind [m/s]        — M2T1NXSLV or M2I1NXASM (v2, optional)
    'v10':   'V10M',     # 10 m northward wind [m/s]       — M2T1NXSLV or M2I1NXASM (v2, optional)
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
# Conservative (area-weighted) regridding
# ===========================================================================

REGRID_METHODS = ('nearest', 'conservative')

# netCDF-C chunk cache per variable, bytes (see InputData.__init__).
NC_CHUNK_CACHE_BYTES = 200 * 1024 * 1024


def _cell_edges(centres, clip=None):
    """Cell edges of a monotonically increasing 1-D cell-centre axis.

    Interior edges are midpoints between neighbouring centres; the two outer
    edges extrapolate half the end spacing. With clip=(lo, hi) the edges are
    clipped, which turns the ERA5 / MERRA-2 pole rows (centres at +-90) into
    half cells, as they are on the sphere.
    """
    c = np.asarray(centres, dtype=np.float64)
    if c.size < 2:
        raise ValueError("Need at least two cell centres to build cell edges")
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])
    return edges


def _interval_overlap(src_edges, tgt_edges, shift=0.0, measure=None):
    """Sparse (n_tgt, n_src) matrix of overlap between two sets of 1-D cells.

    Both edge arrays must be monotonically increasing. `shift` is added to the
    source edges (used for periodic longitude copies). `measure` maps an edge
    coordinate to the measure in which overlaps are counted (sin(lat) for
    latitude so that the overlap is proportional to true spherical area);
    None means plain length. Returns a scipy.sparse.csr_matrix.
    """
    from scipy import sparse

    s = np.asarray(src_edges, dtype=np.float64) + shift
    t = np.asarray(tgt_edges, dtype=np.float64)
    ns, nt = s.size - 1, t.size - 1
    # Candidate source cells for each target cell: those whose upper edge is
    # above the target lower edge and whose lower edge is below the target
    # upper edge. searchsorted gives the index ranges without a Python loop
    # over cells.
    first = np.searchsorted(s[1:], t[:-1], side='right')   # first src with hi > tgt lo
    last = np.searchsorted(s[:-1], t[1:], side='left')     # one past last src with lo < tgt hi
    first = np.minimum(first, ns)
    last = np.clip(last, first, ns)
    counts = last - first
    if counts.sum() == 0:
        return sparse.csr_matrix((nt, ns), dtype=np.float64)
    rows = np.repeat(np.arange(nt), counts)
    cols = np.concatenate([np.arange(a, b) for a, b in zip(first, last) if b > a])
    lo = np.maximum(s[cols], t[rows])
    hi = np.minimum(s[cols + 1], t[rows + 1])
    if measure is None:
        w = hi - lo
    else:
        w = measure(hi) - measure(lo)
    # Discard slivers from floating-point edge noise (e.g. 0.05 deg edges vs
    # 0.25 deg edges that should coincide but differ in the last bit).
    tol = 1e-9 * min(np.min(np.diff(s)), np.min(np.diff(t)))
    keep = (hi - lo) > tol
    M = sparse.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=(nt, ns),
                          dtype=np.float64)
    M.sum_duplicates()
    return M


def _sin_deg(x):
    return np.sin(np.deg2rad(x))


class ConservativeRegridder:
    """First-order conservative (area-weighted) regridding between two
    regular lat/lon cell-centre grids.

    The overlap weights are separable: W[(i,j),(k,l)] = A_lat[i,k] * A_lon[j,l]
    where A_lat holds sin(lat) overlaps (true spherical area) and A_lon plain
    longitude overlaps with periodic wrap. The 2-D weight matrix is never
    formed; the operator is applied as A_lat @ (X @ A_lon.T), which is exact
    and keeps memory small even for a 0.05 deg source (3600 x 7200).

    Weights are normalised by the target-cell extent so that a source grid
    that fully covers a target cell gives row sums of exactly one. Missing
    source values (NaN or < -900) are excluded and the result is normalised
    over the VALID overlap only: a target cell with any valid source gets the
    area mean of its valid part, a target cell with none gets `fill`.

    Fine-to-coarse: area mean of the source cells inside each target cell.
    Coarse-to-fine: piecewise constant (each target cell takes the area mean
    of the source cells it overlaps, i.e. the one it sits in).

    Source latitude may run N to S or S to N; source longitude may be in
    -180..180 or 0..360. The target grid is given by 1-D centre arrays in
    whatever order the meteorology file uses; the output follows that order.
    """

    def __init__(self, src_lon, src_lat, tgt_lon, tgt_lat):
        from scipy import sparse

        src_lon = np.asarray(src_lon, dtype=np.float64)
        src_lat = np.asarray(src_lat, dtype=np.float64)
        tgt_lon = np.asarray(tgt_lon, dtype=np.float64)
        tgt_lat = np.asarray(tgt_lat, dtype=np.float64)

        # Work on increasing axes; remember which axes must be flipped.
        self.flip_src_lat = bool(src_lat[0] > src_lat[-1])
        self.flip_src_lon = bool(src_lon[0] > src_lon[-1])
        self.flip_tgt_lat = bool(tgt_lat[0] > tgt_lat[-1])
        self.flip_tgt_lon = bool(tgt_lon[0] > tgt_lon[-1])
        s_lat = src_lat[::-1] if self.flip_src_lat else src_lat
        s_lon = src_lon[::-1] if self.flip_src_lon else src_lon
        t_lat = tgt_lat[::-1] if self.flip_tgt_lat else tgt_lat
        t_lon = tgt_lon[::-1] if self.flip_tgt_lon else tgt_lon
        for name, ax in (('src_lat', s_lat), ('src_lon', s_lon),
                         ('tgt_lat', t_lat), ('tgt_lon', t_lon)):
            if np.any(np.diff(ax) <= 0):
                raise ValueError(f"{name} is not strictly monotonic")

        self.src_shape = (src_lat.size, src_lon.size)
        self.tgt_shape = (tgt_lat.size, tgt_lon.size)

        # Latitude: overlaps measured in sin(lat), i.e. spherical area.
        s_lat_e = _cell_edges(s_lat, clip=(-90.0, 90.0))
        t_lat_e = _cell_edges(t_lat, clip=(-90.0, 90.0))
        A_lat = _interval_overlap(s_lat_e, t_lat_e, measure=_sin_deg)
        tgt_lat_ext = _sin_deg(t_lat_e[1:]) - _sin_deg(t_lat_e[:-1])

        # Longitude: plain angular overlap with periodic copies of the source.
        s_lon_e = _cell_edges(s_lon)
        t_lon_e = _cell_edges(t_lon)
        src_span = s_lon_e[-1] - s_lon_e[0]
        if src_span > 360.0 + 1e-6:
            raise ValueError(f"Source longitude cells span {src_span:.6f} deg > 360; "
                             "a duplicated wrap column would be double counted")
        A_lon = None
        for k in (-2, -1, 0, 1, 2):
            M = _interval_overlap(s_lon_e, t_lon_e, shift=360.0 * k)
            A_lon = M if A_lon is None else A_lon + M
        tgt_lon_ext = t_lon_e[1:] - t_lon_e[:-1]

        # Normalise by target-cell extent (row sums == 1 where fully covered).
        self.A_lat = (sparse.diags(1.0 / tgt_lat_ext) @ A_lat).tocsr()
        self.A_lon = (sparse.diags(1.0 / tgt_lon_ext) @ A_lon).tocsr()
        self.A_lonT = self.A_lon.T.tocsr()

        # Cell areas (steradian) for diagnostics and conservation checks,
        # in the caller's axis order.
        src_lat_ext = _sin_deg(s_lat_e[1:]) - _sin_deg(s_lat_e[:-1])
        src_lon_ext = s_lon_e[1:] - s_lon_e[:-1]
        self.tgt_area = np.outer(tgt_lat_ext, np.deg2rad(tgt_lon_ext))
        self.src_area = np.outer(src_lat_ext, np.deg2rad(src_lon_ext))
        if self.flip_tgt_lat:
            self.tgt_area = self.tgt_area[::-1, :]
        if self.flip_tgt_lon:
            self.tgt_area = self.tgt_area[:, ::-1]
        if self.flip_src_lat:
            self.src_area = self.src_area[::-1, :]
        if self.flip_src_lon:
            self.src_area = self.src_area[:, ::-1]
        self.tgt_area = np.ascontiguousarray(self.tgt_area)
        self.src_area = np.ascontiguousarray(self.src_area)

        # Denominator for the all-valid case, computed once.
        self._den_all_valid = self._apply_raw(np.ones(self.src_shape))

    def _apply_raw(self, X):
        """A_lat @ X @ A_lon.T for X of shape (..., n_src_lat, n_src_lon)."""
        lead = X.shape[:-2]
        nsl, nslon = self.src_shape
        ntl, ntlon = self.tgt_shape
        X2 = X.reshape(-1, nslon)                        # (T*nsl, nslon)
        Y = X2 @ self.A_lonT                             # dense (T*nsl, ntlon)
        Y = np.asarray(Y).reshape(-1, nsl, ntlon)        # (T, nsl, ntlon)
        Y = np.ascontiguousarray(Y.transpose(1, 0, 2)).reshape(nsl, -1)
        Z = self.A_lat @ Y                               # (ntl, T*ntlon)
        Z = np.asarray(Z).reshape(ntl, -1, ntlon).transpose(1, 0, 2)
        return Z.reshape(lead + (ntl, ntlon))

    def __call__(self, data, fill=np.nan):
        """Regrid `data` of shape (..., n_src_lat, n_src_lon). Returns float64
        of shape (..., n_tgt_lat, n_tgt_lon) in the target axis order."""
        X = np.array(data, dtype=np.float64)   # copy; never modify the caller's array
        if X.shape[-2:] != self.src_shape:
            raise ValueError(f"data shape {X.shape[-2:]} does not match source grid "
                             f"{self.src_shape}")
        if self.flip_src_lat:
            X = X[..., ::-1, :]
        if self.flip_src_lon:
            X = X[..., :, ::-1]
        valid = np.isfinite(X) & (X >= -900.0)
        all_valid = bool(valid.all())
        if not all_valid:
            X = np.where(valid, X, 0.0)
        num = self._apply_raw(X)
        if all_valid:
            den = np.broadcast_to(self._den_all_valid, num.shape)
        else:
            den = self._apply_raw(valid.astype(np.float64))
        covered = den > 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(covered, num / np.where(covered, den, 1.0), fill)
        if self.flip_tgt_lat:
            out = out[..., ::-1, :]
        if self.flip_tgt_lon:
            out = out[..., :, ::-1]
        return np.ascontiguousarray(out)


def conservative_regrid(src_lon, src_lat, data, tgt_lon, tgt_lat, fill=np.nan):
    """One-off conservative regrid; see ConservativeRegridder for the rules."""
    return ConservativeRegridder(src_lon, src_lat, tgt_lon, tgt_lat)(data, fill=fill)


# ===========================================================================
# Input data loader
# ===========================================================================

class InputData:
    """Loads meteorological driving data (ERA5 or MERRA-2), Prigent z0a, MODIS LAI,
    MODIS land cover, and SoilGrids data."""

    def __init__(self, meteo_path, soilgrids_dir, prigent_file, modis_lai_dir,
                 landcover_file=None, meteo_source='era5', brdf_file=None,
                 regrid_method='nearest', dead_dir=None):
        """meteo_path: ERA5 or MERRA-2 file/directory. meteo_source: 'era5' or 'merra2'.
        brdf_file: NetCDF of monthly MODIS BRDF omega_n (scripts/brdf_tif_to_nc.py),
        needed by the albedo-based scheme only.
        regrid_method: 'nearest' (default, nearest neighbour) or 'conservative'
        (area-weighted) for the static inputs regridded onto the meteorology grid.
        dead_dir: directory with the GEOS-Chem DEAD inputs (HEMCO DUST_DEAD,
        GOCART_src_fn.geos.2x25.nc and dst_tibds.geos.2x25.nc), needed by the
        dead_g / dead_z schemes only."""
        self.meteo_source = meteo_source
        self.dead_dir = Path(dead_dir) if dead_dir else None
        if regrid_method not in REGRID_METHODS:
            raise ValueError(f"Unknown regrid_method: {regrid_method!r} "
                             f"(use one of {REGRID_METHODS})")
        self.regrid_method = regrid_method
        self._regridders = {}  # {(src_lat bytes, src_lon bytes): ConservativeRegridder}
        # netCDF chunk cache (v2, 2026-09-24). The monthly meteo files are
        # chunked 186 timesteps deep (e.g. MERRA-2: 186 x 91 x 144, 9.7 MB per
        # chunk, 16 chunks per hourly slab), so with the 16 MB library default
        # every hourly read decompresses the whole chunk row again (about 0.5 s
        # per variable and step). A cache that holds one chunk row per variable
        # makes the next 185 reads free. Numerically neutral. Applies to files
        # opened after this call; the limit is per variable, not per file.
        nc4.set_chunk_cache(size=NC_CHUNK_CACHE_BYTES, nelems=4133, preemption=0.75)
        self.era5_path = self.meteo_path = Path(meteo_path)
        self.soilgrids_dir = Path(soilgrids_dir)
        self.prigent_file = prigent_file
        self.modis_lai_dir = Path(modis_lai_dir)
        self.landcover_file = landcover_file
        self.brdf_file = Path(brdf_file) if brdf_file else None
        self._brdf_cache = {}  # {month: {'omega_n': array on the meteo grid}}
        self.S = None          # DEAD erodibility map on the meteo grid (load_dead_S)

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
            self.lsm = np.where(np.isfinite(disph_raw) & (disph_raw > 0), 1.0, 0.0)
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
        # 10 m winds are optional (v2 wind-driven schemes)
        if 'u10' in self._era5_ds.variables and 'v10' in self._era5_ds.variables:
            fields['u10'] = self._read_era5('u10', tidx=local_tidx)
            fields['v10'] = self._read_era5('v10', tidx=local_tidx)
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
        # 10 m winds are optional (v2 wind-driven schemes; prepare_merra2.py --wind-source)
        if (MERRA2_VARS['u10'] in self._era5_ds.variables
                and MERRA2_VARS['v10'] in self._era5_ds.variables):
            fields['u10'] = self._read_era5(MERRA2_VARS['u10'], tidx=local_tidx)
            fields['v10'] = self._read_era5(MERRA2_VARS['v10'], tidx=local_tidx)
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
        """Regrid a (lat, lon) source field onto the meteorology grid.

        Dispatches on self.regrid_method: 'nearest' is the original nearest
        neighbour path (unchanged), 'conservative' the area-weighted overlap
        method (ConservativeRegridder), with weights cached per source grid.
        Both treat NaN and values < -900 as missing.
        """
        if self.regrid_method == 'conservative':
            return self._regrid_conservative(src_lon, src_lat, data, fill=fill)
        return self._regrid_nearest(src_lon, src_lat, data, fill=fill)

    def _regrid_conservative(self, src_lon, src_lat, data, fill=0.0):
        """Area-weighted regrid; weight matrices cached per source grid."""
        src_lon = np.asarray(src_lon, dtype=np.float64)
        src_lat = np.asarray(src_lat, dtype=np.float64)
        key = (src_lat.tobytes(), src_lon.tobytes())
        rg = self._regridders.get(key)
        if rg is None:
            rg = ConservativeRegridder(src_lon, src_lat, self.lon, self.lat)
            self._regridders[key] = rg
        return rg(data, fill=fill)

    def _regrid_nearest(self, src_lon, src_lat, data, fill=0.0):
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

    def get_brdf(self, dt):
        """MODIS BRDF omega_n for the month of dt, on the meteorology grid.

        Reads the monthly NetCDF written by scripts/brdf_tif_to_nc.py (variable
        omega_n, dims month/lat/lon). Missing and negative values become 0 (no
        sheltering information: ocean or no data), values are capped at 50, and
        the field is regridded with the same method as the other statics.
        """
        month = dt.month if hasattr(dt, 'month') else int(str(dt)[5:7])
        if month in self._brdf_cache:
            return self._brdf_cache[month]
        if self.brdf_file is None:
            raise ValueError("the albedo-based scheme needs --modis-brdf <modis_brdf_YYYY.nc>")
        with nc4.Dataset(self.brdf_file, 'r') as ds:
            brdf_lon = np.array(ds.variables['lon'][:], dtype=np.float64)
            brdf_lat = np.array(ds.variables['lat'][:], dtype=np.float64)
            omega_n = np.array(ds.variables['omega_n'][month - 1, :, :]).astype(np.float64)
        omega_n = np.nan_to_num(omega_n, nan=0.0)
        omega_n = np.clip(omega_n, 0.0, 50.0)
        if omega_n.shape == (self.nlat, self.nlon):
            omega_regrid = omega_n
        else:
            omega_regrid = self._regrid(brdf_lon, brdf_lat, omega_n, fill=0.0)
        result = {'omega_n': omega_regrid}
        self._brdf_cache[month] = result
        valid = omega_regrid > 0
        log.info(f"  BRDF month {month:02d}: omega_n mean={omega_regrid[valid].mean():.2f}, "
                 f"valid={int(valid.sum())} cells")
        return result

    def load_dead_S(self, scheme):
        """Load the DEAD erodibility map S for scheme 'dead_g' (GOC_SRC from
        GOCART_src_fn.geos.2x25.nc) or 'dead_z' (EF_GEO from
        dst_tibds.geos.2x25.nc), regrid it onto the meteorology grid with the
        run's regrid method (nearest or conservative), fill 0 outside the
        data, and keep it in self.S. Time invariant, so it goes to static['S']."""
        if scheme not in DEAD_S_SOURCE:
            raise ValueError(f"no DEAD erodibility source for scheme {scheme!r}")
        if self.dead_dir is None:
            raise ValueError(f"--scheme {scheme} needs --dead-inputs <dir> "
                             "(GEOS-Chem DUST_DEAD directory with "
                             "GOCART_src_fn.geos.2x25.nc and dst_tibds.geos.2x25.nc)")
        fname, varname = DEAD_S_SOURCE[scheme]
        fpath = self.dead_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"DEAD input {fpath} not found")
        log.info(f"DEAD erodibility S: {fpath} [{varname}]")
        with nc4.Dataset(str(fpath), 'r') as ds:
            s_lon = np.array(ds.variables['lon'][:], dtype=np.float64)
            s_lat = np.array(ds.variables['lat'][:], dtype=np.float64)
            S_src = np.array(ds.variables[varname][:], dtype=np.float64)
        if S_src.ndim == 3:
            S_src = S_src[0]
        S_src = np.nan_to_num(S_src, nan=0.0)
        S_src = np.clip(S_src, 0.0, None)
        self.S = self._regrid(s_lon, s_lat, S_src, fill=0.0)
        self.S = np.clip(np.nan_to_num(self.S, nan=0.0), 0.0, None)
        n_pos = int(np.sum(self.S > 0))
        log.info(f"  S ({varname}): source {S_src.shape} max={S_src.max():.4f}; "
                 f"on meteo grid {n_pos} cells > 0, max={self.S.max():.4f}, "
                 f"mean(>0)={self.S[self.S > 0].mean() if n_pos else 0:.4f}")
        return self.S

    def close(self):
        if self._era5_ds:
            self._era5_ds.close()
            self._era5_ds = None


# ===========================================================================
# Dust emission physics — Leung et al. (2023)
# ===========================================================================

def surface_state(met, static, lai):
    """
    Shared surface state for every emission scheme (v2 split, 2026-09-24).

    Steps 1 to 11 of the v1.0 compute_emission(): land and snow mask, air
    density, median particle diameter, dry fluid threshold (Shao and Lu, 2000),
    impact threshold, Fecan et al. (1999) moisture correction, Marticorena and
    Bergametti (1995) rock drag partition, Okin (2008) vegetation drag
    partition, saltation friction velocity, bare-soil fraction and the
    standardized fluid threshold. Every expression is unchanged from v1.0, so
    the result is bitwise identical; the scheme-specific flux is computed
    from this state by flux_kok14() and by the other schemes.

    Parameters
    ----------
    met : dict — ERA5 {zust, swvl1, t2m, d2m, sp, sd, [blh]} or MERRA-2
          {zust, rhoa, swvl1, sd, [blh]}
    static : dict — {lsm, z0a, clay, silt, bulk_den, D_p, z0s, f_erodible}
    lai : ndarray (lat, lon) — MODIS LAI [m²/m²]

    Returns
    -------
    state : dict of ndarray (lat, lon) fields, see the return statement.
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


    return {
        'mask': mask, 'rhoa': rhoa, 'D_p': D_p, 'z0s': z0s,
        'ustar_ft0': ustar_ft0, 'ustar_it': ustar_it,
        'f_moisture': f_moisture, 'ustar_ft_wet': ustar_ft_wet,
        'clay_pct': clay_pct, 'z0a_valid': z0a_valid,
        'f_eff_r': f_eff_r, 'f_eff_v': f_eff_v, 'F_eff': F_eff,
        'wnd_frc_slt': wnd_frc_slt, 'f_bare': f_bare, 'ustar_st': ustar_st,
        'gwc': gwc,
    }


HOLDABLE = ('f_eff_r', 'f_eff_v', 'f_moisture', 'C_d', 'kappa', 'eta')


def flux_kok14(met, static, state, C_tune=0.05, hold=()):
    """
    Kok et al. (2014) emission flux, Leung et al. (2023) formulation, from a
    surface state (v2 split, 2026-09-24). Steps 12 to 16 of the v1.0
    compute_emission(): soil erodibility C_d, fragmentation exponent kappa,
    clay scaling, turbulent intermittency eta (Comola et al., 2019) and the
    flux equation. Expressions unchanged from v1.0.

    hold : sequence of factor names from HOLDABLE, each held at its neutral
        value for a sensitivity ladder (v2, --hold): 'f_eff_r' and 'f_eff_v'
        at 1 (no rock or vegetation drag partition, u*_s recomputed),
        'f_moisture' at 1 (dry threshold, u*_st recomputed), 'C_d' at C_d0
        (constant erodibility), 'kappa' at 0 (no fragmentation exponent),
        'eta' at 1 (no intermittency). With hold=() the code path is the v1.0
        path unchanged.

    Returns
    -------
    F_d : ndarray (lat, lon) — dust emission flux [kg/m²/s]
    diag : dict — diagnostic fields (same keys as v1.0, plus clay_scale)
    """
    C = CONST
    hold = tuple(hold)
    for h in hold:
        if h not in HOLDABLE:
            raise ValueError(f"cannot hold '{h}'; choose from {HOLDABLE}")
    mask = state['mask']
    rhoa = state['rhoa']
    D_p = state['D_p']
    ustar_ft0 = state['ustar_ft0']
    ustar_it = state['ustar_it']
    f_moisture = state['f_moisture']
    ustar_ft_wet = state['ustar_ft_wet']
    clay_pct = state['clay_pct']
    z0a_valid = state['z0a_valid']
    f_eff_r = state['f_eff_r']
    f_eff_v = state['f_eff_v']
    F_eff = state['F_eff']
    wnd_frc_slt = state['wnd_frc_slt']
    f_bare = state['f_bare']
    ustar_st = state['ustar_st']

    if 'f_eff_r' in hold or 'f_eff_v' in hold:
        if 'f_eff_r' in hold:
            f_eff_r = np.ones_like(f_eff_r)
        if 'f_eff_v' in hold:
            f_eff_v = np.ones_like(f_eff_v)
        F_eff = f_eff_r * f_eff_v
        wnd_frc_slt = met['zust'] * F_eff
    if 'f_moisture' in hold:
        f_moisture = np.ones_like(f_moisture)
        ustar_ft_wet = ustar_ft0 * f_moisture
        ustar_st = ustar_ft_wet * np.sqrt(rhoa / C['rho_a0'])

    # ------------------------------------------------------------------
    # 12. Soil erodibility coefficient C_d
    #     C_d = C_d0 × exp(-C_e × (u*_st - u*_st0) / u*_st0)
    # ------------------------------------------------------------------
    C_d = C['C_d0'] * np.exp(-C['C_e'] * (ustar_st - C['ustar_st0']) / C['ustar_st0'])
    if 'C_d' in hold:
        C_d = np.full_like(C_d, C['C_d0'])

    # ------------------------------------------------------------------
    # 13. Fragmentation exponent κ (capped at 3)
    #     κ = C_α × (u*_st - u*_st0) / u*_st0
    # ------------------------------------------------------------------
    kappa = C['C_alpha'] * (ustar_st - C['ustar_st0']) / C['ustar_st0']
    kappa = np.clip(kappa, 0.0, C['kappa_cap'])
    if 'kappa' in hold:
        kappa = np.zeros_like(kappa)

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
    if 'eta' in hold:
        eta = np.ones_like(rhoa)

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
        'F_eff': F_eff, 'kappa': kappa, 'C_d': C_d, 'eta': eta, 'clay_scale': clay_scale,
        'D_p': D_p, 'f_moisture': f_moisture,
    }
    return F_d, diag


# ---------------------------------------------------------------------------
# Albedo-based drag partition scheme (Chappell and Webb, 2016) with MB95
# saltation. Reinstated in v2 from the development tree (2026-09-24); the
# archived January 2006 development run (output_2006_aem) is its regression pin.
# ---------------------------------------------------------------------------
AEM_CONST = {
    # u*_s/U_h as a function of rescaled normalised shadow omega_ns, fit from the
    # Zenodo 5816911 Google Earth Engine implementation (Hennen and Chappell).
    'aem_A':      0.0306,     # amplitude
    'aem_C':      1.1202,     # exponent on omega_ns
    'aem_D':      0.0125,     # denominator in the exponential argument
    'aem_E':      0.0072,     # offset (minimum u*_s/U_h)
    'aem_wn_max': 35.0,       # omega_n cap for rescaling (Marshall 1971 wind-tunnel range)
    'aem_ns_min': 0.0001,     # rescaled omega_ns at a bare bright surface
    'aem_ns_max': 0.1,        # rescaled omega_ns at a rough dark surface
}

AEM_SIZE_BINS = np.array([
    # d [m],       u*_t [m/s]
    [2.0e-6,       2.10],
    [5.0e-6,       1.00],
    [1.0e-5,       0.60],
    [2.2e-5,       0.40],
    [5.2e-5,       0.2127],
    [9.0e-5,       0.22],
    [1.25e-4,      0.25],
    [2.50e-4,      0.31],
    [5.00e-4,      0.40],
    [1.00e-3,      0.55],
])
AEM_UST_T = AEM_SIZE_BINS[:, 1]  # threshold u*_t per bin


def flux_aem(met, static, state, brdf):
    """
    Albedo-based drag partition (Chappell and Webb, 2016) with MB95 saltation
    and the Marticorena (1999) clay sandblasting efficiency, from a surface
    state. No global tuning constant (c = 1).

    Physics chain:
      1. MODIS BRDF omega_n -> rescale -> u*_s/U_h -> u*_s
      2. MB95 saltation: Q = (rho_a/g) u*_s^3 sum_i E_f(d_i), 10 size bins,
         Fecan moisture correction applied to each bin threshold
      3. Vertical flux: F = f_bare Q alpha(clay)

    brdf : dict — {'omega_n': array on the meteo grid} from InputData.get_brdf
    Bare fraction is f_erodible x mask, without the LAI factor of the K14 chain,
    because vegetation sheltering enters through the albedo here.
    """
    C = CONST
    A = AEM_CONST
    mask = state['mask']
    rhoa = state['rhoa']
    f_moisture = state['f_moisture']
    clay_pct = state['clay_pct']

    # 1. AEM drag partition -> u*_s
    omega_n = brdf['omega_n']
    wn_clipped = np.clip(omega_n, 0.0, A['aem_wn_max'])
    omega_ns = (A['aem_ns_min']
                + (A['aem_ns_max'] - A['aem_ns_min'])
                * wn_clipped / A['aem_wn_max'])

    ustar_ratio = (A['aem_A']
                   * np.exp(-(omega_ns ** A['aem_C']) / A['aem_D'])
                   + A['aem_E'])

    if 'u10' in met and 'v10' in met:
        U_10m = np.sqrt(met['u10'] ** 2 + met['v10'] ** 2)
    else:
        z0_fb = np.where(
            np.isfinite(static['z0a']) & (static['z0a'] > 0),
            static['z0a'], 3e-4)
        U_10m = met['zust'] * np.log(10.0 / z0_fb) / C['k_vk']

    wnd_frc_slt = ustar_ratio * U_10m

    # 2. Bare soil fraction
    f_bare = static['f_erodible'] * mask

    # 3. MB95 saltation flux over 10 size bins
    active = (wnd_frc_slt > AEM_UST_T.min()) & (mask > 0.5) & (f_bare > 0)
    F_d = np.zeros_like(rhoa)

    if np.any(active):
        u_s = wnd_frc_slt[active]
        rho_a = rhoa[active]
        f_moist = f_moisture[active]

        E_sum = np.zeros_like(u_s)
        for u_t_bin in AEM_UST_T:
            u_t = u_t_bin * f_moist
            emitting = u_s > u_t
            ratio = np.where(emitting, u_t / u_s, 1.0)
            E_f = np.where(emitting,
                           (1.0 + ratio) * (1.0 - ratio ** 2),
                           0.0)
            E_sum += E_f

        Q = (rho_a / C['g']) * u_s ** 3 * E_sum

        # alpha = 10^(0.134 clay% - 6), Marticorena (1999), clay capped at 20 %
        clay_active = np.clip(clay_pct[active], 0.0, 20.0)
        alpha = 10.0 ** (0.134 * clay_active - 6.0)

        F_d[active] = f_bare[active] * Q * alpha

    F_d = np.nan_to_num(F_d, nan=0.0, posinf=0.0, neginf=0.0)
    F_d = np.clip(F_d, 0.0, None)

    diag = {
        'rhoa': rhoa, 'wnd_frc_slt': wnd_frc_slt,
        'f_bare': f_bare, 'f_moisture': f_moisture,
        'ustar_ratio': ustar_ratio, 'omega_ns': omega_ns,
        'U_10m': U_10m,
    }
    return F_d, diag


# ---------------------------------------------------------------------------
# DEAD scheme (Zender et al., 2003a) as configured in GEOS-Chem / HEMCO.
# Added in v2 (2026-09-24). Reference implementation: HEMCO 3.9.1
# hcox_dustdead_mod.F (the version GEOS-Chem 14.4.1 used); published
# description: Zhang et al. (2025, GMD 18, 6767) Appendix A2, Eqs. A2 to A9.
# ---------------------------------------------------------------------------
DEAD_CONST = {
    'g':           9.80665,   # m s-2, GRV_SFC in the Fortran (Zhang A2 writes 9.81)
    'k_vk':        0.4,       # von Karman, CST_VON_KRM
    'z_ref':       10.0,      # m, HGT_RFR: reference height of the wind
    'z0_mbl':      100.0e-6,  # m, RGH_MMN_MBL: roughness of erodible surfaces (BLM_MBL)
    'wnd_min':     1.0,       # m s-1, WND_MIN_MBL: floor on the 10 m wind (BLM_MBL)
    'D_slt_opt':   75.0e-6,   # m, DMT_SLT_OPT: optimal saltation diameter (IvW82)
    'rho_slt':     2650.0,    # kg m-3, DNS_SLT: density of saltating particles
    'C_slt':       2.61,      # CST_SLT: White (1979) saltation constant
    'gwc_thr_a':   0.17,      # kg kg-1, Fecan threshold intercept (FRC_THR_NCR_WTR_GET)
    'gwc_thr_b':   0.14,      # kg kg-1 per clay fraction, Fecan threshold slope
    'clay_cap':    0.20,      # max clay fraction in the MB95 sandblasting law
    'alpha_unit':  100.0,     # m-1 per cm-1: MB95 Eq. 47 is in cm-1, the Fortran uses m-1
    'rho_water':   1000.0,    # kg m-3, DNS_H2O_LQD_STD (VWC2GWC)
    # Global mass tuning factor T (FLX_MSS_FDG_FCT, "Mass tuning factor" in
    # HEMCO_Config.rc). GEOS-Chem 14.4.1 run/GCClassic/createRunDir.sh sets
    # RUNDIR_DUSTDEAD_TF = 4.7586e-4 (merra2, 2x2.5), 7.8533e-4 (merra2, 4x5),
    # 5.0416e-4 (geosfp, 2x2.5), 8.3286e-4 (geosfp, 4x5) and -999 (undefined)
    # for the native 0.5x0.625 and 0.25x0.3125 grids, where GEOS-Chem reads
    # the offline DEAD emissions of Meng et al. (2021) instead. Those files
    # (HEMCO/OFFLINE_DUST/v2021-08, README of 23 April 2021) carry the
    # MERRA-2 native-grid factor 4.7376e-4, which is the value used here.
    'T_default':   4.7376e-4,
}

# Erodibility map S per DEAD configuration: (file in --dead-inputs, variable).
DEAD_S_SOURCE = {
    'dead_g': ('GOCART_src_fn.geos.2x25.nc', 'GOC_SRC'),   # Ginoux et al. (2001)
    'dead_z': ('dst_tibds.geos.2x25.nc', 'EF_GEO'),        # Zender et al. (2003b) geomorphic
}


def dead_ustar_10m(u10, v10):
    """Friction velocity from the 10 m wind by the neutral log profile
    (BLM_MBL): u* = k max(U10, 1 m/s) / ln(z_ref / z0_mbl), z0_mbl = 1e-4 m.
    Returns (u*, U10 after the floor)."""
    D = DEAD_CONST
    U = np.sqrt(np.asarray(u10, dtype=np.float64) ** 2
                + np.asarray(v10, dtype=np.float64) ** 2)
    U = np.maximum(U, D['wnd_min'])
    return U * D['k_vk'] / np.log(D['z_ref'] / D['z0_mbl']), U


def dead_ustar_ft0(rhoa):
    """Iversen and White (1982) dry fluid threshold at D_p = 75 um
    (WND_FRC_THR_SLT_GET), as a function of air density only.

    Re  = 0.38 + 1331 (100 D_p)^1.56            (D_p in m, 100 D_p in cm)
    B   = 1 + 6e-7 / (rho_p g D_p^2.5)
    Re < 10:  G = -1 + 1.928 Re^0.0922,  u*ft0 = 0.129 sqrt(B / G) sqrt(rho_p g D_p / rho_a)
    Re >= 10: G = 1 - 0.0858 exp(-0.0617 (Re - 10)),
              u*ft0 = 0.12 sqrt(B) G sqrt(rho_p g D_p / rho_a)
    At 75 um Re = 1.02, so the first branch applies.
    """
    D = DEAD_CONST
    Dp = D['D_slt_opt']
    Re = 0.38 + 1331.0 * (100.0 * Dp) ** 1.56
    B = 1.0 + 6.0e-7 / (D['rho_slt'] * D['g'] * Dp ** 2.5)
    if Re < 10.0:
        G = -1.0 + 1.928 * Re ** 0.0922
        tmp1 = 0.129 * 0.129 * B / G
    else:
        G = 1.0 - 0.0858 * np.exp(-0.0617 * (Re - 10.0))
        tmp1 = 0.12 * 0.12 * B * G * G
    dns_fct = D['rho_slt'] * D['g'] * Dp
    return np.sqrt(tmp1) * np.sqrt(dns_fct / np.asarray(rhoa, dtype=np.float64))


def dead_fecan(gwc, f_clay):
    """Fecan et al. (1999) moisture factor as coded in FRC_THR_NCR_WTR_GET:
    w_t = 0.17 + 0.14 f_clay (kg kg-1; the leading f_clay of Fecan Eq. 14 was
    removed in the Fortran on 1999-11-05 'to improve large scale behavior'),
    f_m = sqrt(1 + 1.21 (100 (w - w_t))^0.68) for w > w_t, else 1.
    Returns (f_m, w_t)."""
    D = DEAD_CONST
    gwc = np.asarray(gwc, dtype=np.float64)
    gwc_thr = D['gwc_thr_a'] + D['gwc_thr_b'] * np.asarray(f_clay, dtype=np.float64)
    gwc_thr = np.broadcast_to(gwc_thr, gwc.shape)
    f_m = np.ones_like(gwc)
    wet = gwc > gwc_thr
    f_m[wet] = np.sqrt(1.0 + 1.21 * (100.0 * (gwc[wet] - gwc_thr[wet])) ** 0.68)
    return f_m, gwc_thr


def dead_saltation(rhoa, ustar_s, ustar_ft):
    """White (1979) streamwise saltation flux (FLX_MSS_HRZ_SLT_TTL_WHI79_GET):
    Q_s = C_slt (rho_a / g) u*s^3 (1 - r)(1 + r)^2, r = u*ft / u*s, for u*s > u*ft,
    else 0. Units kg m-1 s-1."""
    D = DEAD_CONST
    rhoa = np.asarray(rhoa, dtype=np.float64)
    ustar_s = np.asarray(ustar_s, dtype=np.float64)
    ustar_ft = np.asarray(ustar_ft, dtype=np.float64)
    Q = np.zeros_like(ustar_s)
    on = ustar_s > ustar_ft
    r = ustar_ft[on] / ustar_s[on]
    Q[on] = (D['C_slt'] * rhoa[on] * ustar_s[on] ** 3
             * (1.0 - r) * (1.0 + r) * (1.0 + r) / D['g'])
    return Q


def dead_alpha(clay_pct):
    """Marticorena and Bergametti (1995) sandblasting efficiency as coded in
    FLX_MSS_VRT_DST_TTL_MAB95_GET: alpha = 100 x 10^(13.4 min(f_clay, 0.20) - 6)
    in m-1 (MB95 Eq. 47 gives cm-1; the factor 100 converts). clay_pct in percent."""
    D = DEAD_CONST
    f_clay = np.minimum(np.asarray(clay_pct, dtype=np.float64) / 100.0, D['clay_cap'])
    return D['alpha_unit'] * 10.0 ** (13.4 * f_clay - 6.0)


def flux_dead(met, static, state, S=None, T=DEAD_CONST['T_default'], clay_pct=None):
    """
    DEAD dust emission scheme (Zender et al., 2003a) as configured in
    GEOS-Chem / HEMCO (hcox_dustdead_mod.F, HEMCO 3.9.1), from a surface state.
    Registered twice: 'dead_g' with the GOCART / Ginoux et al. (2001)
    topographic source function and 'dead_z' with the Zender et al. (2003b)
    geomorphic erodibility; the two differ only in the map S.

    Equation chain (Zhang et al., 2025, Appendix A2, Eqs. A2 to A9; constants
    from the Fortran where the two differ):
      1. u*   = k max(U10, 1 m/s) / ln(10 m / 1e-4 m)            (A3, BLM_MBL)
         u*s  = u*: no drag partition, no Owen effect (both removed from the
         GEOS-Chem code on 2007-01-25).
      2. u*ft0 = Iversen and White (1982) at D_p = 75 um          (A5 to A7)
         Re = 0.38 + 1331 (100 D_p)^1.56 = 1.02; u*ft0 = 0.129 sqrt(B/G)
         sqrt(rho_p g D_p / rho_a), B = 1 + 6e-7/(rho_p g D_p^2.5),
         G = 1.928 Re^0.0922 - 1; 0.2046 m/s at rho_a = 1.225 kg m-3.
      3. w    = rho_water swvl1 / rho_bulk (gravimetric water content, kg kg-1)
         f_m  = sqrt(1 + 1.21 (100 (w - w_t))^0.68) for w > w_t, else 1  (A8)
         w_t  = 0.17 + 0.14 f_clay                    (FRC_THR_NCR_WTR_GET)
         u*ft = u*ft0 f_m                                              (A4)
      4. Q_s  = C_z (rho_a/g) u*s^3 (1 - u*ft/u*s)(1 + u*ft/u*s)^2, C_z = 2.61,
         for u*s > u*ft, else 0                                        (A2)
      5. alpha = 100 x 10^(13.4 min(f_clay, 0.20) - 6)  [m-1]  (MAB95 routine)
      6. F_d  = Q_s f_bare T S alpha                       (DST_MBL assembly)
    T is the global mass tuning factor FLX_MSS_FDG_FCT ("Mass tuning factor"
    in HEMCO_Config.rc); see DEAD_CONST['T_default'] for its provenance.

    Where the published appendix and the Fortran disagree, the Fortran wins
    for constants:
      * g = 9.80665 (Fortran GRV_SFC) rather than 9.81 (A2).
      * Fecan threshold: A9 gives w_t = 0.01 a (17 f_clay + 14 f_clay^2) with
        a = 1/f_clay = 5 by default; the Fortran computes w_t = 0.17 + 0.14
        f_clay, which is A9 with the leading f_clay divided out (a = 1/f_clay
        exactly), so the two agree only when a is read that way.
      * The 10 m wind is floored at 1 m/s (BLM_MBL WND_MIN_MBL), not in A3.
      * alpha carries the factor 100 (cm-1 to m-1) that Leung et al. (2023,
        Eq. 10) and A2 omit in writing phi = 10^(13.4 f_clay - 6).
      * The threshold uses the mid-layer air density in GEOS-Chem; here the
        shared state['rhoa'] (MERRA-2 RHOA or ERA5 surface density) is used.

    Inputs from the shared layer so that every ODEM scheme sees the same
    forcing and statics: rho_a = state['rhoa']; U10 from met['u10'],
    met['v10'] (error if absent); w = state['gwc'] (rho_water swvl1 / bulk_den
    with SoilGrids bulk density, the same expression as surface_state; the
    Fortran VWC2GWC uses rho_bulk = 2650 (1 - VWC_sat) with VWC_sat = 0.489 -
    0.00126 sand_pct); f_clay = static['clay'] / 100 (SoilGrids); snow and
    land through state['mask']; f_bare = state['f_bare'] = f_erodible x
    max(0, 1 - LAI/LAI_thr) x mask, the shared definition. GEOS-Chem instead
    sets the bare fraction LND_FRC_MBL = 1 x LF_DRY x (1 - snow fraction),
    with snow fraction = min(10 SWE / 0.05 m, 1), zero for frozen soil, lakes,
    land ice and wetlands, and does not use VAI when the GOCART source
    function is on. Threshold: the wet fluid threshold only (DEAD has no impact
    threshold); no intermittency factor.

    GEOS-Chem details NOT reproduced, and why:
      * Uniform clay: SOI_TXT_GET overrides the clay map with MSS_FRC_CLY_GLB
        = 0.20 everywhere ("fxm Temporarily"), so GEOS-Chem's alpha is 0.0479
        m-1 and w_t = 0.198 kg kg-1 in every cell. ODEM uses the SoilGrids clay
        so the schemes share statics; pass clay_pct=20.0 to reproduce the
        GEOS-Chem behaviour (used for the diagnosis runs, never by default).
      * Bare fraction from LF_DRY, fractional snow, soil-temperature freeze
        cutoff (LND_FRC_MBL_GET): replaced by the shared f_bare and mask.
      * Bulk density from VWC_sat(sand) (VWC2GWC, HYD_PRP_GET): replaced by
        SoilGrids bulk density, the shared gwc.
      * Mid-layer virtual-temperature air density (DST_MBL): shared rhoa.
      * Size partition into 4 transport bins (FLX_MSS_VRT_DST_PRT): the sum
        of the bins is 0.9188 of the total vertical flux (lognormal overlap of
        the three d'Almeida modes with 0.2 to 12 um); ODEM returns the total.
      * CaCO3 masking (off in GEOS-Chem), the 350 K temperature cap.

    Parameters
    ----------
    S : ndarray (lat, lon) or None; erodibility on the meteo grid. None takes
        static['S'] (set by run_model from --dead-inputs).
    T : float, global mass tuning factor.
    clay_pct : None (use static['clay']) or a float to hold clay uniform.

    Returns
    -------
    F_d : ndarray (lat, lon), kg m-2 s-1 (total vertical dust flux)
    diag : dict with ustar_10m, U_10m, ustar_ft0_iw, ustar_ft_dead,
           f_moisture_dead, gwc, Q_s, alpha, S, f_bare, rhoa, mask
    """
    if S is None:
        if 'S' not in static:
            raise ValueError("flux_dead needs the erodibility map S: pass S= or "
                             "run with --dead-inputs <dir> (static['S'])")
        S = static['S']
    if 'u10' not in met or 'v10' not in met:
        raise ValueError("flux_dead needs the 10 m wind (met['u10'], met['v10']); "
                         "the meteorology file has no U10M/V10M (u10/v10)")
    mask = state['mask']
    rhoa = np.asarray(state['rhoa'], dtype=np.float64)
    f_bare = state['f_bare']
    gwc = state['gwc']
    clay = static['clay'] if clay_pct is None else np.full_like(rhoa, float(clay_pct))
    f_clay = np.asarray(clay, dtype=np.float64) / 100.0

    ustar_10m, U_10m = dead_ustar_10m(met['u10'], met['v10'])
    ustar_ft0_iw = dead_ustar_ft0(rhoa)
    f_moisture_dead, _ = dead_fecan(gwc, f_clay)
    ustar_ft = ustar_ft0_iw * f_moisture_dead
    Q_s = dead_saltation(rhoa, ustar_10m, ustar_ft)
    alpha = dead_alpha(clay)

    S = np.asarray(S, dtype=np.float64)
    active = (mask > 0.5) & (f_bare > 0) & (S > 0) & (Q_s > 0)
    F_d = np.zeros_like(rhoa)
    F_d[active] = Q_s[active] * f_bare[active] * T * S[active] * alpha[active]
    F_d = np.nan_to_num(F_d, nan=0.0, posinf=0.0, neginf=0.0)
    F_d = np.clip(F_d, 0.0, None)

    diag = {
        'rhoa': rhoa, 'mask': mask, 'ustar_10m': ustar_10m, 'U_10m': U_10m,
        'ustar_ft0_iw': ustar_ft0_iw, 'ustar_ft_dead': ustar_ft,
        'f_moisture_dead': f_moisture_dead, 'gwc': gwc,
        'Q_s': Q_s, 'alpha': alpha, 'S': S, 'f_bare': f_bare,
    }
    return F_d, diag


# Scheme registry (v2). Each flux function takes (met, static, state, **kwargs)
# where state comes from surface_state(), and returns (F_d, diag).
SCHEMES = {'k14': flux_kok14, 'aem': flux_aem,
           'dead_g': flux_dead, 'dead_z': flux_dead}
SCHEME_LABEL = {'k14': 'Kok et al. (2014) + Leung et al. (2023)',
                'aem': 'Chappell and Webb (2016) albedo drag partition + MB95 saltation',
                'dead_g': 'DEAD (Zender et al. 2003a) as in GEOS-Chem, GOCART/Ginoux 2001 source function',
                'dead_z': 'DEAD (Zender et al. 2003a) as in GEOS-Chem, Zender 2003b geomorphic erodibility'}
# Per-timestep inputs a scheme needs beyond met/static/state.
SCHEME_STEP_INPUTS = {'aem': lambda inputs, t: {'brdf': inputs.get_brdf(t)}}
# Label of the scheme of the current run, written to output file attributes.
RUN_SCHEME = {'key': 'k14', 'label': SCHEME_LABEL['k14']}


def compute_emission(met, static, lai, C_tune=0.05):
    """
    Compute dust emission flux for a single timestep — Leung et al. (2023).

    Implements Kok et al. (2014) emission with MB95+Okin drag partition.
    Since v2 this is surface_state() followed by flux_kok14(); the result is
    bitwise identical to v1.0 (checked against the archived runs).

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
    state = surface_state(met, static, lai)
    return flux_kok14(met, static, state, C_tune)


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
        ds.source = RUN_SCHEME['label']
        ds.scheme = RUN_SCHEME['key']
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
        ds.source = RUN_SCHEME['label']
        ds.scheme = RUN_SCHEME['key']
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
        ds.source = RUN_SCHEME['label']
        ds.scheme = RUN_SCHEME['key']
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
        'gwc':          ('Gravimetric soil water content',  'kg kg-1'),
        'U_10m':        ('10 m wind speed',                 'm s-1'),
        'ustar_10m':    ('Friction velocity from the 10 m wind (DEAD)', 'm s-1'),
        'ustar_ft0_iw': ('Iversen-White dry threshold at 75 um (DEAD)', 'm s-1'),
        'ustar_ft_dead':('Wet fluid threshold (DEAD)',      'm s-1'),
        'f_moisture_dead': ('Fecan factor, DEAD threshold w_t', '1'),
        'Q_s':          ('White 1979 saltation flux (DEAD)', 'kg m-1 s-1'),
        'alpha':        ('MB95 sandblasting efficiency (DEAD)', 'm-1'),
        'mask':         ('Land and no-snow mask',           '1'),
    }
    STATIC_META = {
        'D_p':       ('Median soil particle diameter',     'm'),
        'z0a':       ('Prigent aeolian roughness z0a',     'm'),
        'clay':      ('Clay fraction',                     'percent'),
        'silt':      ('Silt fraction',                     'percent'),
        'bulk_den':  ('Soil bulk density',                 'kg m-3'),
        'f_erodible':('MODIS erodibility fraction',        '1'),
        'lsm':       ('Land-sea mask',                     '1'),
        'S':         ('DEAD erodibility map S',            '1'),
    }

    with nc4.Dataset(filepath, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', len(lat))
        ds.createDimension('lon', len(lon))

        lat_v = ds.createVariable('lat', 'f8', ('lat',))
        lat_v[:] = lat; lat_v.units = 'degrees_north'
        lon_v = ds.createVariable('lon', 'f8', ('lon',))
        lon_v[:] = lon; lon_v.units = 'degrees_east'

        for name, arr_sum in diag_sums.items():
            if name in static:
                continue  # the static field of that name is written below
            lname, units = DIAG_META.get(name, (name, '1'))
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
        ds.source = RUN_SCHEME['label']
        ds.scheme = RUN_SCHEME['key']
        ds.period = f'{t_start} to {t_end}'
        ds.n_timesteps = nt
        ds.history = f'Created {datetime.now().isoformat()}'


def run_model(meteo_path, soilgrids_dir, prigent_file, modis_lai_dir,
              landcover_file=None, output_dir='output_v2', C_tune=0.05,
              save_diags=False, meteo_source='era5', scheme='k14',
              max_steps=None, brdf_file=None, regrid_method='nearest',
              hold=(), dead_dir=None, dead_tuning=DEAD_CONST['T_default']):
    """Run ODEM — Offline Dust Emission Model.

    meteo_path:      ERA5 or MERRA-2 file/directory.
    meteo_source:    'era5' (default) or 'merra2'
    save_diags:      if True, write odem_diag_mean.nc with time-mean diagnostic fields.
    scheme:          emission scheme key in SCHEMES ('k14' = v1.0 physics).
    max_steps:       if set, stop after this many timesteps (development and
                     regression runs only; the outputs then cover a partial month).
    brdf_file:       monthly MODIS BRDF NetCDF, required by scheme 'aem'.
    hold:            factors of the k14 scheme held at their neutral value
                     (sensitivity ladder; see flux_kok14 and HOLDABLE).
    regrid_method:   'nearest' (default) or 'conservative' for the static inputs
                     (Prigent z0a, SoilGrids, MODIS land cover, MODIS LAI, BRDF, S).
    dead_dir:        GEOS-Chem DUST_DEAD input directory, required by dead_g / dead_z.
    dead_tuning:     DEAD global mass tuning factor T (FLX_MSS_FDG_FCT).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme '{scheme}'; choose from {sorted(SCHEMES)}")
    if scheme == 'aem' and not brdf_file:
        raise ValueError("--scheme aem needs --modis-brdf <modis_brdf_YYYY.nc>")
    hold = tuple(h for h in hold if h)
    if hold and scheme != 'k14':
        raise ValueError("--hold applies to the k14 scheme only")
    if scheme in DEAD_S_SOURCE and not dead_dir:
        raise ValueError(f"--scheme {scheme} needs --dead-inputs <dir> (GEOS-Chem "
                         "DUST_DEAD directory with GOCART_src_fn.geos.2x25.nc and "
                         "dst_tibds.geos.2x25.nc)")
    RUN_SCHEME['key'] = scheme
    RUN_SCHEME['label'] = SCHEME_LABEL[scheme] + (f" [held: {','.join(hold)}]" if hold else '')
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
    log.info(f"  Scheme:     {scheme} ({SCHEME_LABEL.get(scheme, scheme)})")
    if brdf_file:
        log.info(f"  MODIS BRDF: {brdf_file}")
    if hold:
        log.info(f"  Hold:       {', '.join(hold)} at neutral values")
    if scheme in DEAD_S_SOURCE:
        log.info(f"  DEAD inputs: {dead_dir} ({DEAD_S_SOURCE[scheme][1]})")
        log.info(f"  DEAD T:     {dead_tuning:.4e} (global mass tuning factor)")
    log.info(f"  Regrid:     {regrid_method} (static inputs onto the meteo grid)")
    log.info("=" * 60)

    # Load inputs
    inputs = InputData(meteo_path, soilgrids_dir, prigent_file, modis_lai_dir,
                       landcover_file=landcover_file, meteo_source=meteo_source,
                       brdf_file=brdf_file, regrid_method=regrid_method,
                       dead_dir=dead_dir)
    inputs.load()
    if scheme in DEAD_S_SOURCE:
        inputs.load_dead_S(scheme)

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
    if scheme in DEAD_S_SOURCE:
        static['S'] = inputs.S

    log.info(f"  D_p: median={np.nanmedian(D_p)*1e6:.0f} µm, "
             f"range=[{np.nanmin(D_p)*1e6:.0f}, {np.nanmax(D_p)*1e6:.0f}] µm")

    # Run timesteps — accumulate mean rather than storing all timesteps
    os.makedirs(output_dir, exist_ok=True)
    F_d_sum = np.zeros((inputs.nlat, inputs.nlon), dtype=np.float64)
    F_d_max = np.zeros((inputs.nlat, inputs.nlon), dtype=np.float32)

    # Diagnostic accumulators (only allocated if save_diags=True); the key set
    # is whatever the scheme returns, taken from the first timestep.
    diag_sums = {}
    if scheme == 'k14':
        scheme_kwargs = {'C_tune': C_tune, 'hold': hold}
    elif scheme in DEAD_S_SOURCE:
        scheme_kwargs = {'T': float(dead_tuning)}
    else:
        scheme_kwargs = {}
    flux_fn = SCHEMES[scheme]
    step_inputs = SCHEME_STEP_INPUTS.get(scheme, lambda inputs_, t_: {})
    n_steps = inputs.nt if max_steps is None else min(inputs.nt, int(max_steps))
    if n_steps != n_steps:
        log.warning(f"  max_steps: running {n_steps} of {n_steps} timesteps")

    # For monthly output files: group timesteps by month
    monthly_bufs = {}  # {(year,month): [list of F_d arrays]}

    for tidx in range(n_steps):
        t = inputs.times[tidx]
        t_str = str(t)
        if tidx % 8 == 0:
            log.info(f"  [{tidx+1}/{n_steps}] {t_str}")

        met = inputs.get_timestep(tidx)
        lai = inputs.get_modis_lai(t)
        state = surface_state(met, static, lai)
        F_d, diag = flux_fn(met, static, state, **scheme_kwargs, **step_inputs(inputs, t))

        F_d_sum += F_d
        F_d_max = np.maximum(F_d_max, F_d)

        if save_diags:
            if not diag_sums:
                diag_sums = {k: np.zeros((inputs.nlat, inputs.nlon), dtype=np.float64)
                             for k in diag}
            for k in diag_sums:
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
                for k in ('f_eff_r', 'f_eff_v', 'eta', 'ustar_ratio', 'U_10m',
                          'ustar_10m', 'ustar_ft_dead', 'f_moisture_dead', 'alpha'):
                    if k in diag:
                        log.info(f"    {k}: mean={diag[k][active].mean():.3f}")

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

    # Period-mean flux. Always computed — the run summary below reports the
    # global budget from it, so gating the assignment on n_months made every
    # single-month run (including both README quick-start examples) die with
    # UnboundLocalError after the output had already been written.
    F_mean = F_d_sum / n_steps  # kg/m²/s

    # Annual mean output — only written for multi-month runs.
    # Single-month runs would overwrite a valid annual mean with one month's data.
    n_months = len(set((t.year, t.month) for t in inputs.times))
    if n_months > 1:
        out_file = os.path.join(output_dir, 'odem_annual_mean.nc')
        _write_annual_mean(out_file, inputs.lon, inputs.lat, F_mean, F_d_max,
                           inputs.times[0], inputs.times[-1], n_steps)
    else:
        log.info("  Skipping annual mean file (single-month run)")

    # Diagnostic output
    if save_diags:
        diag_file = os.path.join(output_dir, 'odem_diag_mean.nc')
        _write_diag_mean(diag_file, inputs.lon, inputs.lat, diag_sums,
                         static, n_steps, inputs.times[0], inputs.times[-1])

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
    p.add_argument('--regrid', choices=REGRID_METHODS, default='nearest',
                   help='How static inputs (Prigent z0a, SoilGrids, MODIS land cover, '
                        'MODIS LAI) are put on the meteo grid: nearest neighbour '
                        '(default) or conservative area-weighted')

    p.add_argument('--scheme', default='k14', choices=sorted(SCHEMES),
                   help='Emission scheme (default k14 = Kok 2014 / Leung 2023, the v1.0 physics)')
    p.add_argument('--max-steps', type=int, default=None,
                   help='Development only: stop after N timesteps (partial-month output)')
    p.add_argument('--hold', default='',
                   help='k14 sensitivity ladder: comma-separated factors held at their '
                        f'neutral value, from {",".join(HOLDABLE)}')
    p.add_argument('--modis-brdf', default=None,
                   help='Monthly MODIS BRDF omega_n NetCDF (scripts/brdf_tif_to_nc.py); '
                        'required for --scheme aem')
    p.add_argument('--dead-inputs', default=None,
                   help='GEOS-Chem DUST_DEAD input directory (GOCART_src_fn.geos.2x25.nc, '
                        'dst_tibds.geos.2x25.nc); required for --scheme dead_g / dead_z')
    p.add_argument('--dead-tuning', type=float, default=DEAD_CONST['T_default'],
                   help='DEAD global mass tuning factor T (FLX_MSS_FDG_FCT); default '
                        f"{DEAD_CONST['T_default']:.4e} (GEOS-Chem MERRA-2 native grid)")
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
        scheme=args.scheme,
        max_steps=args.max_steps,
        brdf_file=args.modis_brdf,
        regrid_method=args.regrid,
        hold=tuple(h.strip() for h in args.hold.split(',') if h.strip()),
        dead_dir=args.dead_inputs,
        dead_tuning=args.dead_tuning,
    )


if __name__ == '__main__':
    main()
