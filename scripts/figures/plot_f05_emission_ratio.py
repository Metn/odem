#!/usr/bin/env python3
"""
ODEM Paper — Figure 5: ERA5/MERRA-2 emission ratio map
========================================================
Map of the ratio ODEM-ERA5 / ODEM-M2 at each grid cell, showing where
ERA5 produces more or less emission than MERRA-2.

ERA5 emission is regridded to the M2 grid before computing the ratio.

Output: papers/odem_gmd/figures/f05.pdf + f05.png

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


def make_figure():
    apply_gmd_style()

    # Load both annual means
    print("  Loading ODEM-ERA5 annual mean...")
    ds_e = nc4.Dataset(str(ERA5_ANNUAL))
    lat_e, lon_e = ds_e["lat"][:], ds_e["lon"][:]
    em_e = np.array(ds_e["F_d_mean"][:])
    ds_e.close()

    print("  Loading ODEM-M2 annual mean...")
    ds_m = nc4.Dataset(str(M2_ANNUAL))
    lat_m, lon_m = ds_m["lat"][:], ds_m["lon"][:]
    em_m = np.array(ds_m["F_d_mean"][:])
    ds_m.close()

    # Regrid ERA5 to M2 grid
    from scipy.interpolate import RegularGridInterpolator
    if lat_e[0] > lat_e[-1]:
        lat_e = lat_e[::-1]
        em_e = em_e[::-1, :]

    interp = RegularGridInterpolator(
        (lat_e, lon_e), em_e,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    lon2d, lat2d = np.meshgrid(lon_m, lat_m)
    em_e_regrid = interp((lat2d, lon2d))

    # Compute ratio — only where both have emission
    min_flux = 1e-13  # threshold to avoid noise
    valid = (em_e_regrid > min_flux) & (em_m > min_flux)
    ratio = np.full_like(em_m, np.nan)
    ratio[valid] = em_e_regrid[valid] / em_m[valid]
    ratio = np.ma.masked_where(~valid, ratio)

    print(f"    Ratio range: {np.ma.min(ratio):.2f} — {np.ma.max(ratio):.2f}")
    print(f"    Median ratio: {np.ma.median(ratio):.2f}")

    # Plot
    fig, ax = plt.subplots(
        figsize=(7.2, 3.5),
        subplot_kw={"projection": GMD_PROJ},
    )

    # Diverging colormap centred at ratio=1 (log scale)
    norm = mcolors.LogNorm(vmin=0.2, vmax=5.0)

    im = ax.pcolormesh(
        lon_m, lat_m, ratio,
        transform=GMD_DATA_CRS,
        cmap="RdBu_r", norm=norm, rasterized=True,
    )
    add_map_features(ax)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.06, shrink=0.7, aspect=30)
    cbar.set_label("Emission ratio (ODEM-ERA5 / ODEM-M2)", fontsize=8)
    cbar.set_ticks([0.2, 0.5, 1.0, 2.0, 5.0])
    cbar.set_ticklabels(["0.2", "0.5", "1.0", "2.0", "5.0"])
    cbar.ax.tick_params(labelsize=7)
    # Mark ratio=1 line
    cbar.ax.axvline(1.0, color="0.2", linewidth=0.8)

    ax.set_title("ODEM-ERA5 / ODEM-M2 annual emission ratio", fontsize=9, pad=5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f05.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 5: ERA5/M2 emission ratio map")
    print("=" * 50)
    make_figure()
    print("\nDone!")
