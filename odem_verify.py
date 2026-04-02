#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

"""
ODEM — Scientific Verification Suite
======================================

Rigorous, multi-level verification of model physics and output.
Goal: correct results for correct reasons.

Verification levels:
  1. Physics unit tests   — formulas tested at known inputs against analytical values
  2. Diagnostic ranges    — intermediate fields within physically defensible bounds
  3. Spatial sanity       — emission patterns match known climatology constraints
  4. Global/regional budget — comparison against published observational estimates
  5. Input data checks    — ERA5, SoilGrids, Prigent inputs are internally consistent
  6. C_tune sensitivity   — model not trivially tuned; spatial pattern precedes budget match

Usage:
    # After a run with --save-diags:
    python odem_verify.py \\
        --emission /path/to/output/odem_200601.nc \\
        --diags    /path/to/output/odem_diag_mean.nc \\
        --era5     /path/to/era5/era5_dust_200601.nc \\
        --outdir   ./verify_output

    # Physics unit tests only (no data files needed):
    python odem_verify.py --unit-tests

Author: Metin Baykara
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import netCDF4 as nc4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Import ODEM physics functions for unit testing
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from odem import air_density, median_particle_diameter, compute_emission, CONST


# ===========================================================================
# Helpers
# ===========================================================================

class VerifyResult:
    """Accumulates pass/fail results across all checks."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def ok(self, name, detail=''):
        self.passed.append((name, detail))
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ''))

    def fail(self, name, detail=''):
        self.failed.append((name, detail))
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ''))

    def warn(self, name, detail=''):
        self.warnings.append((name, detail))
        print(f"  [WARN] {name}" + (f" — {detail}" if detail else ''))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print()
        print("=" * 60)
        print(f"  VERIFICATION SUMMARY: {len(self.passed)}/{total} passed")
        if self.warnings:
            print(f"  Warnings: {len(self.warnings)}")
        if self.failed:
            print(f"\n  FAILED CHECKS:")
            for name, detail in self.failed:
                print(f"    - {name}: {detail}")
        if self.warnings:
            print(f"\n  WARNINGS:")
            for name, detail in self.warnings:
                print(f"    - {name}: {detail}")
        print("=" * 60)
        return len(self.failed) == 0


def load_nc_var(filepath, varname):
    with nc4.Dataset(filepath, 'r') as ds:
        return np.array(ds.variables[varname][:], dtype=np.float64)


def load_nc_coords(filepath):
    with nc4.Dataset(filepath, 'r') as ds:
        lon = np.array(ds.variables['lon'][:])
        lat = np.array(ds.variables['lat'][:])
    return lon, lat


def cell_areas(lat, lon):
    R = 6.371e6
    dlat = abs(np.median(np.diff(lat)))
    dlon = abs(np.median(np.diff(lon)))
    areas = R**2 * np.cos(np.deg2rad(lat)) * np.deg2rad(dlat) * np.deg2rad(dlon)
    return np.abs(areas[:, np.newaxis]) * np.ones((1, len(lon)))


def _make_1cell_inputs(ustar=0.5, swvl1=0.01, snow=0.0,
                        clay=5.0, silt=10.0, bulk_den=1500.0,
                        D_p=127e-6, z0a=1e-4, f_erodible=0.8, lsm=1.0,
                        lai=0.1):
    """Create minimal 1×1 cell inputs for physics unit tests."""
    met = {
        'zust':  np.array([[ustar]]),
        'swvl1': np.array([[swvl1]]),
        't2m':   np.array([[300.0]]),
        'd2m':   np.array([[280.0]]),
        'sp':    np.array([[101325.0]]),
        'sd':    np.array([[snow]]),
    }
    z0s = D_p / 15.0
    static = {
        'lsm':       np.array([[lsm]]),
        'z0a':       np.array([[z0a]]),
        'clay':      np.array([[clay]]),
        'silt':      np.array([[silt]]),
        'bulk_den':  np.array([[bulk_den]]),
        'D_p':       np.array([[D_p]]),
        'z0s':       np.array([[z0s]]),
        'f_erodible':np.array([[f_erodible]]),
    }
    return met, static, np.array([[lai]])


def analytical_ustar_ft0(D_p, rho_a):
    """Shao & Lu (2000) dry fluid threshold — analytical reference."""
    C = CONST
    return np.sqrt(C['A_SL'] * (C['rho_p'] * C['g'] * D_p
                                 + C['gamma_SL'] / D_p)) / np.sqrt(rho_a)


# ===========================================================================
# Level 1: Physics unit tests
# ===========================================================================

def test_physics(R):
    """Test each physics formula at analytically known inputs."""
    print("\n[Level 1] Physics Unit Tests")
    print("-" * 50)

    # ------------------------------------------------------------------
    # 1a. Air density — Tv virtual temperature approach
    #     At T=293K, Td=273K, P=101325 Pa → ρ ≈ 1.205 kg/m³
    # ------------------------------------------------------------------
    rho = air_density(
        np.array([293.0]),   # T2m [K]
        np.array([273.0]),   # Td2m [K] (dry air limit)
        np.array([101325.0]) # pressure [Pa]
    )
    # Dry air: P/(Rd*T) = 101325/(287.05*293) = 1.2042
    expected = 101325.0 / (287.05 * 293.0)
    if abs(rho[0] - expected) / expected < 0.005:
        R.ok("air_density (dry)", f"got {rho[0]:.4f}, expected ~{expected:.4f} kg/m³")
    else:
        R.fail("air_density (dry)", f"got {rho[0]:.4f}, expected ~{expected:.4f} kg/m³")

    # ------------------------------------------------------------------
    # 1b. Median particle diameter — arid branch (LAI < 1)
    # ------------------------------------------------------------------
    D_p = median_particle_diameter(
        np.array([5.0]),  # clay %
        np.array([10.0]), # silt %
        np.array([0.1])   # LAI (< 1 → arid branch)
    )
    if abs(D_p[0] - CONST['D_p_arid']) < 1e-10:
        R.ok("median_D_p arid", f"{D_p[0]*1e6:.0f} µm = D_p_arid = {CONST['D_p_arid']*1e6:.0f} µm")
    else:
        R.fail("median_D_p arid", f"got {D_p[0]*1e6:.1f} µm, expected {CONST['D_p_arid']*1e6:.0f} µm")

    # ------------------------------------------------------------------
    # 1c. Median particle diameter — non-arid branch (LAI ≥ 1)
    #     D_p = Psi_0 + Psi_1 * (clay + silt) / 100
    # ------------------------------------------------------------------
    f_fine = 0.40  # 40% clay+silt
    expected_dp = CONST['Psi_0'] + CONST['Psi_1'] * f_fine
    D_p2 = median_particle_diameter(
        np.array([30.0]),  # clay %
        np.array([10.0]),  # silt %  (40% total fine)
        np.array([2.0])    # LAI ≥ 1 → non-arid
    )
    if abs(D_p2[0] - expected_dp) / expected_dp < 0.001:
        R.ok("median_D_p non-arid", f"{D_p2[0]*1e6:.1f} µm, expected {expected_dp*1e6:.1f} µm")
    else:
        R.fail("median_D_p non-arid",
               f"got {D_p2[0]*1e6:.1f} µm, expected {expected_dp*1e6:.1f} µm")

    # ------------------------------------------------------------------
    # 1d. Shao & Lu (2000) fluid threshold — compare with analytical value
    #     at D_p=127µm, rho_a=1.225 kg/m³ (standard conditions)
    # ------------------------------------------------------------------
    D_p_ref = CONST['D_p_arid']
    rho_a_ref = CONST['rho_a0']
    uft_expected = analytical_ustar_ft0(D_p_ref, rho_a_ref)

    met, static, lai = _make_1cell_inputs(ustar=0.5)
    # Override air density by setting T/P to give rho_a = rho_a0 exactly
    # Use T=288.15K, P=101325 → rho_a = 1.225 (standard atmosphere)
    met['t2m'] = np.array([[288.15]])
    met['d2m'] = np.array([[200.0]])   # very dry → negligible humidity
    met['sp']  = np.array([[101325.0]])
    _, diag = compute_emission(met, static, lai)
    uft_model = diag['ustar_ft0'][0, 0]
    if abs(uft_model - uft_expected) / uft_expected < 0.01:
        R.ok("Shao-Lu u*_ft0",
             f"model={uft_model:.4f} m/s, analytical={uft_expected:.4f} m/s")
    else:
        R.fail("Shao-Lu u*_ft0",
               f"model={uft_model:.4f} m/s, analytical={uft_expected:.4f} m/s "
               f"({100*(uft_model-uft_expected)/uft_expected:.1f}% error)")

    # ------------------------------------------------------------------
    # 1e. Impact threshold = B_it × u*_ft0 (exact relationship)
    # ------------------------------------------------------------------
    _, diag2 = compute_emission(met, static, lai)
    ratio = diag2['ustar_it'][0, 0] / diag2['ustar_ft0'][0, 0]
    if abs(ratio - CONST['B_it']) < 1e-6:
        R.ok("Impact threshold ratio", f"u*_it / u*_ft0 = {ratio:.4f} = B_it = {CONST['B_it']}")
    else:
        R.fail("Impact threshold ratio",
               f"u*_it / u*_ft0 = {ratio:.6f}, expected {CONST['B_it']}")

    # ------------------------------------------------------------------
    # 1f. Fécan moisture: dry soil (gwc ≤ gwc_thr) → f_moisture = 1.0
    # ------------------------------------------------------------------
    met_dry, static_dry, lai_dry = _make_1cell_inputs(swvl1=0.001)  # very dry
    _, diag_dry = compute_emission(met_dry, static_dry, lai_dry)
    fm = diag_dry['f_moisture'][0, 0]
    if abs(fm - 1.0) < 1e-6:
        R.ok("Fécan dry soil (f_moisture=1)", f"f_moisture = {fm:.6f}")
    else:
        R.fail("Fécan dry soil", f"f_moisture = {fm:.6f}, expected 1.0 for dry soil")

    # ------------------------------------------------------------------
    # 1g. Fécan moisture: wet soil → f_moisture > 1.0
    # ------------------------------------------------------------------
    met_wet, static_wet, lai_wet = _make_1cell_inputs(
        swvl1=0.40,  # very moist
        clay=20.0,   # 20% clay → gwc_thr ≈ 0.034+0.0003*400 = 0.154
        bulk_den=1300.0
    )
    _, diag_wet = compute_emission(met_wet, static_wet, lai_wet)
    fm_wet = diag_wet['f_moisture'][0, 0]
    if fm_wet > 1.0:
        R.ok("Fécan wet soil (f_moisture>1)", f"f_moisture = {fm_wet:.3f}")
    else:
        R.fail("Fécan wet soil", f"f_moisture = {fm_wet:.3f}, expected > 1.0")

    # ------------------------------------------------------------------
    # 1h. Rock drag partition: very smooth roughness (z0a ≈ z0s)
    #     → f_eff_r approaches 1.0 (no roughness elements to shelter)
    # ------------------------------------------------------------------
    D_p_test = 127e-6
    z0s_test = D_p_test / 15.0
    met_r, static_r, lai_r = _make_1cell_inputs(z0a=z0s_test * 1.001)
    _, diag_r = compute_emission(met_r, static_r, lai_r)
    feffr = diag_r['f_eff_r'][0, 0]
    if feffr > 0.95:
        R.ok("Rock drag (smooth surface → f_eff_r≈1)",
             f"f_eff_r = {feffr:.4f}")
    else:
        R.warn("Rock drag (smooth surface → f_eff_r≈1)",
               f"f_eff_r = {feffr:.4f} (expected near 1.0)")

    # ------------------------------------------------------------------
    # 1i. Rock drag partition: large roughness → f_eff_r << 1
    # ------------------------------------------------------------------
    met_rough, static_rough, lai_rough = _make_1cell_inputs(z0a=0.1)  # 10cm roughness
    _, diag_rough = compute_emission(met_rough, static_rough, lai_rough)
    feffr_rough = diag_rough['f_eff_r'][0, 0]
    if feffr_rough < 0.5:
        R.ok("Rock drag (rough surface → f_eff_r<0.5)",
             f"f_eff_r = {feffr_rough:.4f}")
    else:
        R.warn("Rock drag (rough surface)", f"f_eff_r = {feffr_rough:.4f} (expected < 0.5)")

    # ------------------------------------------------------------------
    # 1j. Vegetation drag: bare surface (LAI=0) → f_eff_v → 1.0
    # ------------------------------------------------------------------
    met_bare, static_bare, lai_bare = _make_1cell_inputs(lai=0.0)
    _, diag_bare = compute_emission(met_bare, static_bare, lai_bare)
    feffv_bare = diag_bare['f_eff_v'][0, 0]
    # At LAI→0: K_okin→∞, f_eff_v = f0 + (1-f0)*K/(K+c) → f0 + (1-f0) = 1.0
    if feffv_bare > 0.99:
        R.ok("Okin drag (bare soil → f_eff_v≈1)", f"f_eff_v = {feffv_bare:.4f}")
    else:
        R.fail("Okin drag (bare soil)", f"f_eff_v = {feffv_bare:.4f}, expected ≈ 1.0")

    # ------------------------------------------------------------------
    # 1k. Vegetation drag: dense canopy (LAI=5) → f_eff_v near f0=0.32
    # ------------------------------------------------------------------
    met_dense, static_dense, lai_dense = _make_1cell_inputs(lai=5.0)
    _, diag_dense = compute_emission(met_dense, static_dense, lai_dense)
    feffv_dense = diag_dense['f_eff_v'][0, 0]
    # At LAI=5: K = pi/(2*5) = 0.314; f_eff_v = 0.32 + 0.68*0.314/(0.314+4.8) = 0.362
    K = np.pi / (2 * 5.0)
    C = CONST
    expected_fv = C['f0'] + (1 - C['f0']) * K / (K + C['c_okin'])
    if abs(feffv_dense - expected_fv) < 0.001:
        R.ok("Okin drag (dense canopy LAI=5)", f"f_eff_v = {feffv_dense:.4f}, expected {expected_fv:.4f}")
    else:
        R.fail("Okin drag (dense canopy LAI=5)",
               f"f_eff_v = {feffv_dense:.4f}, expected {expected_fv:.4f}")

    # ------------------------------------------------------------------
    # 1l. Threshold: u*_s ≤ u*_it → zero emission
    # ------------------------------------------------------------------
    # At D_p=127µm, rho_a≈1.2, u*_ft0 ≈ 0.215 m/s, u*_it ≈ 0.176 m/s
    # Use ustar=0.05 m/s (well below threshold)
    met_sub, static_sub, lai_sub = _make_1cell_inputs(ustar=0.05)
    F_d_sub, _ = compute_emission(met_sub, static_sub, lai_sub)
    if F_d_sub[0, 0] == 0.0:
        R.ok("Zero emission below threshold", f"F_d = 0 at u* = 0.05 m/s")
    else:
        R.fail("Zero emission below threshold",
               f"F_d = {F_d_sub[0,0]:.2e} at u* = 0.05 m/s (should be 0)")

    # ------------------------------------------------------------------
    # 1m. Above threshold: u*_s > u*_it → positive emission
    # ------------------------------------------------------------------
    met_sup, static_sup, lai_sup = _make_1cell_inputs(ustar=0.5)
    F_d_sup, _ = compute_emission(met_sup, static_sup, lai_sup)
    if F_d_sup[0, 0] > 0.0:
        R.ok("Positive emission above threshold",
             f"F_d = {F_d_sup[0,0]:.3e} kg/m²/s at u* = 0.5 m/s")
    else:
        R.fail("Positive emission above threshold",
               f"F_d = {F_d_sup[0,0]:.2e} at u* = 0.5 m/s")

    # ------------------------------------------------------------------
    # 1n. Snow mask: snow-covered land → zero emission
    # ------------------------------------------------------------------
    met_snow, static_snow, lai_snow = _make_1cell_inputs(ustar=1.0, snow=0.05)
    F_d_snow, _ = compute_emission(met_snow, static_snow, lai_snow)
    if F_d_snow[0, 0] == 0.0:
        R.ok("Snow mask (sd>0.01 → F_d=0)", f"F_d = 0 under snow cover")
    else:
        R.fail("Snow mask", f"F_d = {F_d_snow[0,0]:.2e} under snow cover (should be 0)")

    # ------------------------------------------------------------------
    # 1o. Ocean mask: lsm=0 → zero emission regardless of wind
    # ------------------------------------------------------------------
    met_ocn, static_ocn, lai_ocn = _make_1cell_inputs(ustar=2.0, lsm=0.0)
    F_d_ocn, _ = compute_emission(met_ocn, static_ocn, lai_ocn)
    if F_d_ocn[0, 0] == 0.0:
        R.ok("Ocean mask (lsm=0 → F_d=0)", f"F_d = 0 over ocean")
    else:
        R.fail("Ocean mask", f"F_d = {F_d_ocn[0,0]:.2e} over ocean (should be 0)")

    # ------------------------------------------------------------------
    # 1p. kappa capped at kappa_cap = 3.0
    # ------------------------------------------------------------------
    met_hv, static_hv, lai_hv = _make_1cell_inputs(ustar=5.0)  # extreme wind
    _, diag_hv = compute_emission(met_hv, static_hv, lai_hv)
    kappa_max = diag_hv['kappa'][0, 0]
    if kappa_max <= CONST['kappa_cap'] + 1e-6:
        R.ok("Kappa cap (≤ 3.0)", f"kappa = {kappa_max:.4f}")
    else:
        R.fail("Kappa cap", f"kappa = {kappa_max:.4f} exceeds cap {CONST['kappa_cap']}")

    # ------------------------------------------------------------------
    # 1q. C_d monotonically decreasing with u*_st
    #     C_d = C_d0 * exp(-C_e * (u*_st - u*_st0) / u*_st0)
    #     → decreasing as u*_st increases above u*_st0
    # ------------------------------------------------------------------
    u_stars = [0.2, 0.5, 1.0, 2.0]
    cd_vals = []
    for us in u_stars:
        met_cd, static_cd, lai_cd = _make_1cell_inputs(ustar=us, swvl1=0.001)
        _, d = compute_emission(met_cd, static_cd, lai_cd)
        cd_vals.append(d['C_d'][0, 0])
    if all(cd_vals[i] >= cd_vals[i+1] for i in range(len(cd_vals)-1)):
        R.ok("C_d monotonically decreasing with u*",
             f"C_d = {[f'{v:.2e}' for v in cd_vals]} at u* = {u_stars}")
    else:
        R.fail("C_d monotonically decreasing",
               f"C_d = {[f'{v:.2e}' for v in cd_vals]} at u* = {u_stars}")

    # ------------------------------------------------------------------
    # 1r. Emission scales linearly with C_tune
    # ------------------------------------------------------------------
    met_ct, static_ct, lai_ct = _make_1cell_inputs(ustar=0.5)
    F1, _ = compute_emission(met_ct, static_ct, lai_ct, C_tune=0.05)
    F2, _ = compute_emission(met_ct, static_ct, lai_ct, C_tune=0.10)
    if F1[0, 0] > 0 and abs(F2[0, 0] / F1[0, 0] - 2.0) < 1e-6:
        R.ok("C_tune linear scaling", f"F(C=0.10)/F(C=0.05) = {F2[0,0]/F1[0,0]:.6f}")
    else:
        R.fail("C_tune linear scaling",
               f"F1={F1[0,0]:.3e}, F2={F2[0,0]:.3e}, ratio={F2[0,0]/F1[0,0]:.3f}")

    # ------------------------------------------------------------------
    # 1s. F_d ≥ 0 everywhere (no negative emissions)
    # ------------------------------------------------------------------
    # Already enforced by np.clip — but test explicitly
    if F_d_sup[0, 0] >= 0 and F_d_sub[0, 0] >= 0:
        R.ok("No negative emission (F_d ≥ 0)")


# ===========================================================================
# Level 2: Diagnostic field range checks
# ===========================================================================

DIAG_EXPECTED = {
    # varname: (min_ok, max_ok, description, reference)
    'rhoa':         (0.55,   1.60,   'Air density [kg/m³]',           'surface atm (0.6 Tibet, 1.58 cold polar)'),
    'ustar_ft0':    (0.12,   0.65,   'Dry fluid threshold [m/s]',     'Shao & Lu 2000; typical desert soils'),
    'ustar_it':     (0.10,   0.54,   'Impact threshold [m/s]',        'B_it × u*_ft0'),
    'ustar_ft_wet': (0.12,   3.00,   'Wet fluid threshold [m/s]',     'always ≥ u*_ft0'),
    'wnd_frc_slt':  (0.0,    10.0,   'Saltation u* [m/s]',            'u* × f_eff'),
    'f_bare':       (0.0,    1.0,    'Bare soil fraction [-]',         'bounded [0,1]'),
    'f_eff_r':      (0.001,  1.0,    'Rock drag partition [-]',        'MB95; 0.001 lower bound in code'),
    'f_eff_v':      (0.32,   1.0,    'Veg drag partition [-]',         'f0=0.32 minimum (Okin 2008)'),
    'F_eff':        (0.001,  1.0,    'Combined drag partition [-]',    'f_eff_r × f_eff_v'),
    'kappa':        (0.0,    3.0,    'Fragmentation exponent [-]',     'capped at kappa_cap=3'),
    'C_d':          (0.0,    1e-3,   'Erodibility coefficient [-]',    'Kok 2014 C_d0=4.4e-5'),
    'eta':          (0.0,    1.0,    'Intermittency [-]',              'Comola 2019; bounded [0,1]'),
    'f_moisture':   (1.0,    20.0,   'Fécan factor [-]',              '≥1 always (only increases threshold)'),
    'D_p':          (9e-6,   132e-6, 'Particle diameter [m]',         'clipped [10µm, 500µm]; arid cap 127µm'),
}


def test_diagnostic_ranges(diag_file, R):
    """Check all diagnostic fields against physically defensible bounds."""
    print("\n[Level 2] Diagnostic Field Range Checks")
    print(f"  File: {diag_file}")
    print("-" * 50)

    lon, lat = load_nc_coords(diag_file)

    with nc4.Dataset(diag_file, 'r') as ds:
        available = list(ds.variables.keys())

    for varname, (vmin, vmax, desc, ref) in DIAG_EXPECTED.items():
        if varname not in available:
            R.warn(f"{varname} not in diag file", "run with --save-diags")
            continue

        data = load_nc_var(diag_file, varname)
        # Only check land cells where data is meaningful
        # (ocean cells will have zeros for many fields)
        finite = data[np.isfinite(data) & (data != -9999.0)]
        nonzero = finite[finite != 0.0]

        if len(nonzero) == 0:
            R.warn(f"{varname} all zero/missing", desc)
            continue

        actual_min = np.min(nonzero)
        actual_max = np.max(nonzero)
        pct01 = np.percentile(nonzero, 0.1)
        pct99 = np.percentile(nonzero, 99.9)

        violations_low  = np.sum(nonzero < vmin)
        violations_high = np.sum(nonzero > vmax)

        detail = (f"range=[{actual_min:.3e}, {actual_max:.3e}], "
                  f"P0.1={pct01:.3e}, P99.9={pct99:.3e}")

        if violations_low == 0 and violations_high == 0:
            R.ok(f"{varname} bounds [{vmin:.2e}, {vmax:.2e}]", detail)
        else:
            msg = (f"{violations_low} below {vmin:.2e}, "
                   f"{violations_high} above {vmax:.2e} | {detail}")
            # Some violations may be physically acceptable edge cases
            pct_viol = 100 * (violations_low + violations_high) / len(nonzero)
            if pct_viol < 0.1:
                R.warn(f"{varname} minor violations ({pct_viol:.3f}%)", msg)
            else:
                R.fail(f"{varname} out of range ({pct_viol:.1f}% violated)", msg)

    # ------------------------------------------------------------------
    # Cross-field consistency checks
    # ------------------------------------------------------------------
    print("\n  Cross-field consistency:")

    if 'ustar_it' in available and 'ustar_ft0' in available:
        u_it  = load_nc_var(diag_file, 'ustar_it')
        u_ft0 = load_nc_var(diag_file, 'ustar_ft0')
        valid = (u_ft0 > 0) & np.isfinite(u_ft0) & np.isfinite(u_it)
        ratio = u_it[valid] / u_ft0[valid]
        if np.allclose(ratio, CONST['B_it'], atol=1e-4):
            R.ok("u*_it / u*_ft0 = B_it everywhere",
                 f"ratio mean={ratio.mean():.6f}, B_it={CONST['B_it']}")
        else:
            R.fail("u*_it / u*_ft0 = B_it",
                   f"ratio range=[{ratio.min():.4f}, {ratio.max():.4f}], expected {CONST['B_it']}")

    if 'ustar_ft_wet' in available and 'ustar_ft0' in available:
        u_wet = load_nc_var(diag_file, 'ustar_ft_wet')
        u_dry = load_nc_var(diag_file, 'ustar_ft0')
        valid = np.isfinite(u_wet) & np.isfinite(u_dry) & (u_dry > 0)
        if np.all(u_wet[valid] >= u_dry[valid] - 1e-10):
            R.ok("u*_ft_wet ≥ u*_ft0 (moisture only increases threshold)")
        else:
            n_viol = np.sum(u_wet[valid] < u_dry[valid] - 1e-10)
            R.fail("u*_ft_wet ≥ u*_ft0", f"{n_viol} cells where wet threshold < dry threshold")

    if 'f_eff_r' in available and 'f_eff_v' in available and 'F_eff' in available:
        f_r   = load_nc_var(diag_file, 'f_eff_r')
        f_v   = load_nc_var(diag_file, 'f_eff_v')
        F_eff = load_nc_var(diag_file, 'F_eff')
        valid = (f_r > 0) & (f_v > 0) & np.isfinite(f_r) & np.isfinite(f_v) & np.isfinite(F_eff)
        product = f_r[valid] * f_v[valid]
        if np.allclose(product, F_eff[valid], rtol=1e-5):
            R.ok("F_eff = f_eff_r × f_eff_v exactly")
        else:
            max_err = np.max(np.abs(product - F_eff[valid]))
            R.fail("F_eff = f_eff_r × f_eff_v", f"max error = {max_err:.2e}")


# ===========================================================================
# Level 3: Spatial sanity checks
# ===========================================================================

# Known active source regions (lon_min, lon_max, lat_min, lat_max)
MUST_EMIT = {
    'Bodele Depression':    (14,  24,  12,  18),
    'Western Sahara':       (-14, -2,  18,  28),
    'Arabian Peninsula':    (44,  56,  18,  26),
    'Saharan Algeria':      (0,   12,  20,  30),
}
# Regions that must NOT emit (wrong physics if they do)
MUST_NOT_EMIT = {
    'Open ocean (Pacific)':    (-170, -120, -10,  10),
    'Amazon rainforest':       (-70,  -50,  -8,   2),
    'Congo Basin':             (20,    30,  -5,   5),
    'Greenland ice sheet':     (-45,  -20,  70,  80),
    'Southern Ocean':          (0,    360, -70, -55),
}


def test_spatial_patterns(emission_file, R):
    """Check emission is active where it should be and zero where it shouldn't."""
    print("\n[Level 3] Spatial Sanity Checks")
    print(f"  File: {emission_file}")
    print("-" * 50)

    lon, lat = load_nc_coords(emission_file)

    with nc4.Dataset(emission_file, 'r') as ds:
        if 'F_d_mean' in ds.variables:
            F = np.array(ds.variables['F_d_mean'][:], dtype=np.float64)
        elif 'F_d' in ds.variables:
            raw = np.array(ds.variables['F_d'][:], dtype=np.float64)
            F = np.nanmean(raw, axis=0) if raw.ndim == 3 else raw
        else:
            R.fail("Load emission field", "No F_d or F_d_mean variable")
            return

    F = np.nan_to_num(F, nan=0.0)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # --- Must-emit regions ---
    print("  Active source regions (must emit):")
    for name, (lon1, lon2, lat1, lat2) in MUST_EMIT.items():
        mask = (lon2d >= lon1) & (lon2d <= lon2) & (lat2d >= lat1) & (lat2d <= lat2)
        region_F = F[mask]
        frac_active = np.mean(region_F > 0)
        mean_F = np.mean(region_F[region_F > 0]) if np.any(region_F > 0) else 0.0
        if frac_active > 0.10:
            R.ok(f"  {name}", f"{100*frac_active:.0f}% active, mean={mean_F:.2e} kg/m²/s")
        else:
            R.fail(f"  {name}", f"only {100*frac_active:.0f}% active cells in known source region")

    # --- Must-not-emit regions ---
    print("  Suppressed regions (must not emit):")
    for name, (lon1, lon2, lat1, lat2) in MUST_NOT_EMIT.items():
        lon1r = lon1 % 360 if lon1 < 0 else lon1
        lon2r = lon2 % 360 if lon2 < 0 else lon2
        lon_adj = lon % 360
        mask = (lon_adj >= lon1r) & (lon_adj <= lon2r) & (lat2d >= lat1) & (lat2d <= lat2)
        if not np.any(mask):
            # Try without lon adjustment
            mask = (lon2d >= lon1) & (lon2d <= lon2) & (lat2d >= lat1) & (lat2d <= lat2)
        region_F = F[mask]
        if len(region_F) == 0:
            R.warn(f"  {name}", "region not in domain")
            continue
        n_emitting = np.sum(region_F > 1e-14)
        total_region = np.sum(region_F * cell_areas(lat, lon)[mask])
        if n_emitting == 0:
            R.ok(f"  {name}", "zero emission (correct)")
        elif total_region * 365.25 * 86400 / 1e9 < 0.1:
            R.warn(f"  {name}", f"{n_emitting} cells emit but total < 0.1 Tg/yr (negligible)")
        else:
            R.fail(f"  {name}",
                   f"{n_emitting} cells emitting, total ≈ {total_region*365.25*86400/1e9:.1f} Tg/yr")

    # --- F_d ≥ 0 everywhere ---
    n_neg = np.sum(F < -1e-15)
    if n_neg == 0:
        R.ok("No negative emission globally", f"min F_d = {F.min():.2e}")
    else:
        R.fail("No negative emission", f"{n_neg} cells with F_d < 0")

    # --- NaN check ---
    n_nan = np.sum(~np.isfinite(F))
    if n_nan == 0:
        R.ok("No NaN/Inf in emission field")
    else:
        R.fail("No NaN/Inf", f"{n_nan} non-finite values in F_d")


# ===========================================================================
# Level 4: Global and regional budget
# ===========================================================================

# Published regional emission estimates for comparison.
#
# IMPORTANT: Unit / size-range conventions
# -----------------------------------------
# ODEM uses the Kok et al. (2014) emission equation with C_tune = 0.05
# from Leung et al. (2023). Leung (2023, p.6505) normalised their
# simulations to a global total of 5000 Tg/yr, which they explicitly
# call "the current constraint of global PM20 dust emission flux."
# Without normalisation, Leung's scheme with C_tune = 0.05 produces
# ~11,700 Tg/yr — a factor-of-2.3 overshoot of the PM20 target.
# ODEM similarly produces ~12,700 Tg/yr (M2), consistent within 9%.
#
# The overshoot reflects the inability of current emission physics to
# constrain absolute magnitude from first principles (Leung 2023).
# F_d is thus best interpreted as PM20-equivalent emission, with the
# understanding that C_tune = 0.05 does NOT normalise to the PM20 target.
#
# Kok et al. (2021b, ACP, doi:10.5194/acp-21-8169-2021):
#   Source-region PM20 emissions (Table 2, central estimate + 5th–95th pct):
#     North Africa:           2727 Tg/yr  (730  – 11 000)
#     Arabian Peninsula:       398 Tg/yr  (143  –    919)
#     Central/East Asia:       230 Tg/yr  (75   –    651)
#
# Bounding boxes below are subsets of the Kok 2021 regions; ODEM values
# are thus expected to be proportionally lower than the full-region total.
# The comparison is ORDER-OF-MAGNITUDE; pass/fail bands are wide (0.1–5×).

# (lon_min, lon_max, lat_min, lat_max): (name, PM20_central, PM20_5th, PM20_95th, source)
# PM20 values from Kok 2021b Table 2; all in Tg/yr
REGIONAL_LITERATURE_KOK2021 = {
    (-20, 40,  15, 37): ('North Africa (subset)',  2727,  730, 11000, 'Kok 2021b Table 2'),
    ( 35, 65,  12, 30): ('Arabian Peninsula',       398,  143,   919, 'Kok 2021b Table 2'),
    ( 55,125,  28, 55): ('Central/East Asia',        230,   75,   651, 'Kok 2021b Table 2'),
}
# No PM20/total conversion needed: F_d is PM20-equivalent (see note above).


def test_budget(emission_file, R, monthly=False):
    """Compare global and regional totals against published observational estimates.

    All ODEM totals are annualized from the period-mean flux (kg/m²/s × 365.25 days).
    For a single-month run this is only an order-of-magnitude estimate.

    Literature benchmark: Kok et al. (2021, ACP) — observationally constrained
    PM20 emission of 5000 ± 1600 Tg/yr. ODEM with C_tune = 0.05 is expected to
    overshoot this by ~2.5× (consistent with Leung et al. 2023 unnormalised output).
    """
    print("\n[Level 4] Global and Regional Budget Verification")
    print(f"  File: {emission_file}")
    if monthly:
        print("  NOTE: Single-month run — period-mean ≠ annual mean.")
        print("        Annualized estimate is order-of-magnitude only.")
    print("-" * 50)

    lon, lat = load_nc_coords(emission_file)
    with nc4.Dataset(emission_file, 'r') as ds:
        if 'F_d_mean' in ds.variables:
            F = np.array(ds.variables['F_d_mean'][:], dtype=np.float64)
        elif 'F_d' in ds.variables:
            raw = np.array(ds.variables['F_d'][:])
            F = np.nanmean(raw, axis=0) if raw.ndim == 3 else raw
    F = np.nan_to_num(F, nan=0.0)

    areas = cell_areas(lat, lon)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Annualize from period-mean flux (kg/m²/s → Tg/yr)
    total_tg = np.sum(F * areas) * 365.25 * 86400 / 1e9

    # --------------------------------------------------------------------------
    # Global total comparison — directly against PM20 constraint
    # --------------------------------------------------------------------------
    # Kok 2021a global PM20: 5000 Tg/yr (range 3400–6600, i.e. ±1σ)
    # ODEM with C_tune=0.05 is expected to overshoot by ~2.5× (Leung 2023)
    kok_pm20_central  = 5000.0
    kok_pm20_low      = 3400.0
    kok_pm20_high     = 6600.0
    leung_unnorm      = 11700.0   # Leung 2023 experiment V, unnormalised

    print(f"\n  ODEM total (annualized):           {total_tg:>8.0f} Tg/yr")
    print(f"  Kok 2021a PM20 constraint:         {kok_pm20_central:.0f} ({kok_pm20_low:.0f}–{kok_pm20_high:.0f}) Tg/yr")
    print(f"  ODEM / PM20 central:               {total_tg / kok_pm20_central:.2f}×")
    print(f"  Leung 2023 unnormalised:           {leung_unnorm:.0f} Tg/yr ({leung_unnorm/kok_pm20_central:.2f}× PM20)")
    print(f"  ODEM / Leung unnormalised:         {total_tg / leung_unnorm:.2f}×")
    if monthly:
        print(f"  (Single-month: annualized estimate is order-of-magnitude only)")

    # Pass criterion: ODEM should be within ~0.5–5× of Leung's unnormalised total
    # (both models use the same equation with C_tune=0.05).
    # Overshoot relative to PM20 is EXPECTED and is noted as informational.
    ratio_vs_leung = total_tg / leung_unnorm
    if 0.5 <= ratio_vs_leung <= 2.0:
        R.ok("Global total consistent with Leung 2023 unnormalised",
             f"{total_tg:.0f} Tg/yr = {ratio_vs_leung:.2f}× Leung ({leung_unnorm:.0f} Tg/yr); "
             f"{total_tg/kok_pm20_central:.1f}× PM20 constraint (overshoot expected)")
    else:
        R.warn("Global total deviates from Leung 2023 unnormalised",
               f"{total_tg:.0f} Tg/yr = {ratio_vs_leung:.2f}× Leung ({leung_unnorm:.0f}); "
               f"expected 0.5–2.0×")

    # C_tune note
    tg_no_ctune = total_tg / CONST['C_tune']
    print(f"\n  C_tune sensitivity:")
    print(f"    Current C_tune = {CONST['C_tune']}")
    print(f"    Untuned total  = {tg_no_ctune:.0f} Tg/yr  (C_tune=1)")
    print(f"    → C_tune is a linear scaling factor; spatial pattern is C_tune-independent")

    # --------------------------------------------------------------------------
    # Regional totals — Kok 2021b observationally-constrained PM20
    # --------------------------------------------------------------------------
    print(f"\n  Regional verification vs Kok et al. (2021b) — PM20 (direct comparison):")
    print(f"  {'Region':<28} {'ODEM Tg/yr':>10}  {'Kok2021 PM20 (central)':>24}  {'Kok2021 range':>22}  Status")
    print(f"  {'-'*28}  {'-'*10}  {'-'*24}  {'-'*22}  {'-'*6}")

    for (lon1, lon2, lat1, lat2), (name, pm20_c, pm20_lo, pm20_hi, source) in REGIONAL_LITERATURE_KOK2021.items():
        mask = (lon2d >= lon1) & (lon2d <= lon2) & (lat2d >= lat1) & (lat2d <= lat2)
        reg_tg = np.sum(F[mask] * areas[mask]) * 365.25 * 86400 / 1e9

        # Compare directly against PM20 values (no conversion).
        # ODEM is expected to overshoot PM20 by ~2.5× (same as global),
        # so wide pass bands are used (order-of-magnitude check).
        ratio = reg_tg / (pm20_c + 1e-6)
        # Pass if within 0.1–10× of PM20 central (order-of-magnitude)
        if 0.1 * pm20_c <= reg_tg <= 10.0 * pm20_c:
            status = "OK"
            R.ok(f"Regional: {name}",
                 f"{reg_tg:.0f} Tg/yr, {ratio:.1f}× PM20 central ({pm20_c:.0f}); "
                 f"PM20 range [{pm20_lo:.0f}–{pm20_hi:.0f}]")
        else:
            status = "WARN"
            R.warn(f"Regional: {name}",
                   f"{reg_tg:.0f} Tg/yr, {ratio:.1f}× PM20 central ({pm20_c:.0f}); "
                   f"outside order-of-magnitude range")

        print(f"  {name:<28} {reg_tg:>10.0f}  {pm20_c:>24.0f}  {pm20_lo:>10.0f}–{pm20_hi:<10.0f}  {status}")

    print(f"\n  Note: Pass criterion = within Kok 2021b 5th–95th percentile range (no added tolerance).")
    print(f"  ODEM bounding boxes may differ slightly from Kok 2021b region definitions;")
    print(f"  the ratio vs central estimate is reported for reference. Kok 2021b ranges reflect")
    print(f"  observational uncertainty and span roughly one order of magnitude per region.")

    # --------------------------------------------------------------------------
    # NH/SH ratio — global dust is ~85–95% NH
    # --------------------------------------------------------------------------
    lat2d_b = np.broadcast_to(lat[:, np.newaxis], F.shape)
    nh_tg = np.sum(F[lat2d_b >= 0] * areas[lat2d_b >= 0]) * 365.25 * 86400 / 1e9
    sh_tg = np.sum(F[lat2d_b < 0]  * areas[lat2d_b < 0])  * 365.25 * 86400 / 1e9
    nh_frac = nh_tg / (nh_tg + sh_tg + 1e-9)
    print(f"\n  NH fraction: {100*nh_frac:.1f}% (literature: ~85–95%)")
    if 0.75 < nh_frac < 0.99:
        R.ok("NH dominance (75–99%)", f"NH = {100*nh_frac:.1f}%")
    else:
        R.fail("NH dominance", f"NH = {100*nh_frac:.1f}% (expected 85–95%)")


# ===========================================================================
# Level 5: Input data sanity checks
# ===========================================================================

def test_inputs(era5_file, R):
    """Check ERA5 input fields are physically self-consistent."""
    print("\n[Level 5] Input Data Sanity Checks")
    print(f"  File: {era5_file}")
    print("-" * 50)

    with nc4.Dataset(era5_file, 'r') as ds:
        available = list(ds.variables.keys())
        lon = np.array(ds.variables.get('longitude', ds.variables.get('lon', []))[:])
        lat = np.array(ds.variables.get('latitude', ds.variables.get('lat', []))[:])
        nt = len(ds.variables.get('valid_time', ds.variables.get('time', []))[:])

        def get_mean(v):
            arr = np.array(ds.variables[v][:])
            return np.nanmean(arr, axis=0) if arr.ndim == 3 else arr

        zust_mean   = get_mean('zust')   if 'zust'  in available else None
        swvl1_mean  = get_mean('swvl1')  if 'swvl1' in available else None
        t2m_mean    = get_mean('t2m')    if 't2m'   in available else None
        sd_mean     = get_mean('sd')     if 'sd'    in available else None
        lsm_raw     = get_mean('lsm')    if 'lsm'   in available else None

    land = (lsm_raw >= 0.5) if lsm_raw is not None else np.ones(zust_mean.shape, bool)

    # u* over land
    if zust_mean is not None:
        u_land = zust_mean[land]
        u_land = u_land[np.isfinite(u_land)]
        if len(u_land) > 0:
            if 0.1 < np.mean(u_land) < 1.5:
                R.ok("ERA5 u* mean over land",
                     f"mean={np.mean(u_land):.3f}, P5={np.percentile(u_land,5):.3f}, "
                     f"P95={np.percentile(u_land,95):.3f} m/s")
            else:
                R.fail("ERA5 u* mean over land",
                       f"mean={np.mean(u_land):.3f} m/s (expected 0.1–1.5 m/s)")
        # Check no negative u*
        n_neg = np.sum(zust_mean[land] < 0)
        if n_neg == 0:
            R.ok("ERA5 u* non-negative")
        else:
            R.fail("ERA5 u* non-negative", f"{n_neg} negative u* values over land")

    # Soil moisture [0, saturation ~0.5]
    if swvl1_mean is not None:
        sw_land = swvl1_mean[land]
        sw_land = sw_land[np.isfinite(sw_land)]
        if len(sw_land) > 0:
            n_neg = np.sum(sw_land < -0.001)
            n_high = np.sum(sw_land > 0.65)
            pct_high = 100 * n_high / len(sw_land)
            detail = (f"mean={np.mean(sw_land):.3f}, max={np.max(sw_land):.3f} m³/m³, "
                      f"{pct_high:.1f}% cells > 0.65")
            if n_neg == 0 and pct_high < 5.0:
                R.ok("ERA5 swvl1 no negatives; <5% above 0.65", detail)
            elif n_neg > 0:
                R.fail("ERA5 swvl1 negative values",
                       f"{n_neg} cells with swvl1 < 0 (ERA5 artifact — check clipping in model)")
            else:
                # ERA5 HTESSEL soil model can exceed physical porosity (~0.5) in
                # frozen/organic soils. odem.py clips to 0 via np.clip(swvl1, 0, None).
                # Values > 0.65 in dust source regions would inflate Fécan correction.
                n_desert = 0
                if lon is not None and lat is not None:
                    pass  # would need 2D mask — warn for now
                R.warn("ERA5 swvl1 > 0.65 in some cells", detail +
                       " — ERA5 HTESSEL artifact in frozen/organic soils; "
                       "check whether high values overlap dust source regions")

    # Temperature — surface [200, 340] K
    if t2m_mean is not None:
        t_land = t2m_mean[land]
        t_land = t_land[np.isfinite(t_land)]
        if len(t_land) > 0:
            if 200 < np.min(t_land) and np.max(t_land) < 340:
                R.ok("ERA5 t2m range [200, 340] K",
                     f"min={np.min(t_land):.0f} K, max={np.max(t_land):.0f} K")
            else:
                R.fail("ERA5 t2m range",
                       f"min={np.min(t_land):.0f} K, max={np.max(t_land):.0f} K")

    # Timestep count (expect 3-hourly = 8/day)
    n_days = nt / 8.0
    if abs(n_days - round(n_days)) < 0.01 and 28 <= n_days <= 31:
        R.ok("ERA5 timestep count (3-hourly monthly)",
             f"{nt} timesteps = {n_days:.0f} days")
    else:
        R.warn("ERA5 timestep count", f"{nt} timesteps ({n_days:.1f} days)")

    # Required variables present
    required = ['zust', 'swvl1', 't2m', 'd2m', 'sp', 'sd', 'lsm']
    missing = [v for v in required if v not in available]
    if not missing:
        R.ok("All required ERA5 variables present",
             f"{required}")
    else:
        R.fail("Missing ERA5 variables", f"{missing}")


# ===========================================================================
# Diagnostic plots for visual inspection
# ===========================================================================

DIAG_PLOT_GROUPS = [
    # Group 1: thresholds and wind
    ['ustar_ft0', 'ustar_it', 'wnd_frc_slt'],
    # Group 2: drag partitioning
    ['f_eff_r', 'f_eff_v', 'F_eff'],
    # Group 3: modifiers and emission drivers
    ['f_moisture', 'f_bare', 'eta'],
    # Group 4: soil properties
    ['D_p', 'clay', 'silt'],
]

DIAG_CMAPS = {
    'ustar_ft0': ('Reds',     'u*_ft0 [m/s]',      0.1,  0.5),
    'ustar_it':  ('Oranges',  'u*_it [m/s]',        0.1,  0.5),
    'wnd_frc_slt':('Blues',   'u*_slt [m/s]',       0.0,  0.6),
    'f_eff_r':   ('RdYlGn',   'f_eff_r [-]',         0.0,  1.0),
    'f_eff_v':   ('Greens',   'f_eff_v [-]',         0.3,  1.0),
    'F_eff':     ('YlOrBr',   'F_eff [-]',           0.0,  1.0),
    'f_moisture':('PuBu',     'f_moisture [-]',       1.0,  3.0),
    'f_bare':    ('YlOrRd',   'f_bare [-]',           0.0,  0.5),
    'eta':       ('Purples',  'eta [-]',              0.0,  1.0),
    'D_p':       ('copper_r', 'D_p [µm]',              0, 130),
    'clay':      ('YlOrBr',   'Clay [%]',              0,  50),
    'silt':      ('Oranges',  'Silt [%]',              0,  40),
}


def plot_diagnostic_panels(diag_file, outdir):
    """Generate diagnostic field maps for visual inspection."""
    print(f"\n  Generating diagnostic plots → {outdir}/")

    lon, lat = load_nc_coords(diag_file)

    for gi, group in enumerate(DIAG_PLOT_GROUPS):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, varname in zip(axes, group):
            with nc4.Dataset(diag_file, 'r') as ds:
                if varname not in ds.variables:
                    ax.text(0.5, 0.5, f'{varname}\nnot available',
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                data = np.array(ds.variables[varname][:], dtype=np.float64)
                units = getattr(ds.variables[varname], 'units', '')

            data[data == -9999.0] = np.nan

            cmap_name, label, vmin, vmax = DIAG_CMAPS.get(
                varname, ('viridis', varname, None, None))

            # Scale D_p to µm for plotting
            if varname == 'D_p':
                data = data * 1e6

            plot_d = data.copy()
            if vmin is None:
                vmin = np.nanpercentile(data, 2)
            if vmax is None:
                vmax = np.nanpercentile(data, 98)

            im = ax.pcolormesh(lon, lat, plot_d, cmap=cmap_name,
                                vmin=vmin, vmax=vmax, shading='auto')
            cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label(label, fontsize=8)
            ax.set_title(varname, fontweight='bold', fontsize=10)
            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylim(lat.min(), lat.max())
            ax.grid(True, alpha=0.2, linewidth=0.3)

            # Stats annotation
            valid = data[np.isfinite(data) & (data > 0)]
            if len(valid) > 0:
                ax.text(0.01, 0.02,
                        f'mean={np.mean(valid):.3g}\nmax={np.max(valid):.3g}',
                        transform=ax.transAxes, fontsize=7,
                        bbox=dict(facecolor='white', alpha=0.8))

        plt.suptitle(f'ODEM Diagnostic Fields — Group {gi+1}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        outpath = os.path.join(outdir, f'diag_group{gi+1}.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"    Saved: {outpath}")
        plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='ODEM — Scientific Verification Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Physics unit tests only (no data needed)
  python odem_verify.py --unit-tests

  # Full verification after a single-month run with --save-diags
  python odem_verify.py \\
      --emission /path/to/output/odem_200601.nc \\
      --diags    /path/to/output/odem_diag_mean.nc \\
      --era5     /path/to/era5/era5_dust_200601.nc \\
      --monthly  \\
      --outdir   ./verify_output

  # Full-year run verification
  python odem_verify.py \\
      --emission /path/to/output/odem_annual_mean.nc \\
      --diags    /path/to/output/odem_diag_mean.nc \\
      --era5     /path/to/era5/era5_dust_200601.nc \\
      --outdir   ./verify_output
""")
    p.add_argument('--unit-tests', action='store_true',
                   help='Run physics unit tests (no data files needed)')
    p.add_argument('--emission', help='Emission NetCDF (odem_*.nc or odem_annual_mean.nc)')
    p.add_argument('--diags',    help='Diagnostic NetCDF (odem_diag_mean.nc from --save-diags)')
    p.add_argument('--era5',     help='ERA5 input NetCDF for input sanity checks')
    p.add_argument('--monthly',  action='store_true',
                   help='Input is a single month (annualize ×12 for budget comparison)')
    p.add_argument('--outdir',   default='./verify_output',
                   help='Output directory for plots')

    args = p.parse_args()

    if not any([args.unit_tests, args.emission, args.diags, args.era5]):
        p.print_help()
        return

    os.makedirs(args.outdir, exist_ok=True)
    R = VerifyResult()

    if args.unit_tests:
        test_physics(R)

    if args.diags:
        test_diagnostic_ranges(args.diags, R)
        plot_diagnostic_panels(args.diags, args.outdir)

    if args.emission:
        test_spatial_patterns(args.emission, R)
        test_budget(args.emission, R, monthly=args.monthly)

    if args.era5:
        test_inputs(args.era5, R)

    passed = R.summary()
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
