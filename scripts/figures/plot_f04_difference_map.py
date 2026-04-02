#!/usr/bin/env python3
"""
ODEM Paper — Figure 4: Emission difference map (ERA5 − MERRA-2)
================================================================
Single-panel map showing the difference in annual mean dust emission
between ODEM-ERA5 and ODEM-M2, highlighting where the choice of
meteorological forcing has the largest impact.

Uses diverging colormap (blue = M2 higher, red = ERA5 higher).

Output: papers/odem_gmd/figures/f04.pdf + f04.png

Author: Metin Baykara
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from pathlib import Path
import numpy as np
import netCDF4 as nc4
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmd_style import apply_gmd_style, GMD_PROJ, GMD_DATA_CRS, add_map_features, add_panel_label

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = Path(os.environ.get("ODEM_FIG_DIR", "./figures"))

_data_env = os.environ.get("ODEM_DATA")
if not _data_env:
    raise FileNotFoundError(
        "Set ODEM_DATA environment variable to your data directory, e.g.:
"
        "  export ODEM_DATA=/path/to/dust_model_data"
    )
DATA_ROOT = Path(_data_env)

ERA5_ANNUAL = DATA_ROOT / "output_2006_era5_hourly" / "odem_annual_mean.nc"
M2_ANNUAL   = DATA_ROOT / "output_2006_merra2" / "odem_annual_mean.nc"


def load_annual(path):
    ds = nc4.Dataset(str(path))
    lat, lon = ds["lat"][:], ds["lon"][:]
    data = np.ma.filled(ds["F_d_mean"][:], 0.0)
    ds.close()
    return lat, lon, data


def regrid_to_coarser(lat_fine, lon_fine, data_fine, lat_coarse, lon_coarse):
    """Regrid fine-resolution data to coarse grid via bilinear interpolation."""
    interp = RegularGridInterpolator(
        (lat_fine, lon_fine), data_fine,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    lon_grid, lat_grid = np.meshgrid(lon_coarse, lat_coarse)
    return interp((lat_grid, lon_grid))


def make_figure():
    apply_gmd_style()

    if not ERA5_ANNUAL.exists():
        print(f"  ERROR: {ERA5_ANNUAL} not found")
        return
    if not M2_ANNUAL.exists():
        print(f"  ERROR: {M2_ANNUAL} not found — run ODEM-M2 first")
        return

    print("  Loading ODEM-ERA5...")
    lat_e, lon_e, flux_e = load_annual(ERA5_ANNUAL)
    print("  Loading ODEM-M2...")
    lat_m, lon_m, flux_m = load_annual(M2_ANNUAL)

    # Regrid ERA5 (0.25°) to MERRA-2 grid (0.5°×0.625°) for differencing
    print("  Regridding ERA5 to MERRA-2 grid...")
    flux_e_regridded = regrid_to_coarser(lat_e, lon_e, flux_e, lat_m, lon_m)

    diff = flux_e_regridded - flux_m  # positive = ERA5 higher

    # Convert to more readable units: g m-2 yr-1
    sec_per_yr = 3600 * 24 * 365.25
    diff_g = diff * sec_per_yr * 1000  # kg m-2 s-1 → g m-2 yr-1

    # Mask where both are near zero (non-source regions)
    combined = flux_e_regridded + flux_m
    diff_g = np.ma.masked_where(combined < 1e-14, diff_g)

    # Symmetric diverging colormap
    vmax = np.percentile(np.abs(diff_g.compressed()), 98)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(
        figsize=(7.2, 3.5),
        subplot_kw={"projection": GMD_PROJ},
    )

    im = ax.pcolormesh(
        lon_m, lat_m, diff_g,
        transform=GMD_DATA_CRS,
        cmap="RdBu_r",  # red = ERA5 higher, blue = M2 higher
        norm=norm,
        rasterized=True,
    )

    add_map_features(ax)
    ax.set_title("Annual mean emission difference (ODEM-ERA5 minus ODEM-M2)", fontsize=9)

    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal",
        fraction=0.06, pad=0.06, aspect=30, shrink=0.7,
    )
    cbar.set_label(r"$\Delta F_d$ (g m$^{-2}$ yr$^{-1}$)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f04.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 4: Emission difference map (ERA5 - M2)")
    print("=" * 58)
    make_figure()
    print("\nDone!")
