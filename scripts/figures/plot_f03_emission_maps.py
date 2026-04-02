#!/usr/bin/env python3
"""
ODEM Paper — Figure 3: Annual mean dust emission maps
======================================================
Three-panel map comparing annual mean dust emission flux:
  (a) ODEM-ERA5   (0.25°, this study)
  (b) ODEM-M2     (0.5°×0.625°, this study)
  (c) MERRA-2 DUEM (0.5°×0.625°, online GOCART)

All fluxes in kg m⁻² s⁻¹, log-scale colorbar.

Output: papers/odem_gmd/figures/f03.pdf + f03.png

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

# Add scripts dir to path for gmd_style
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmd_style import apply_gmd_style, GMD_PROJ, GMD_DATA_CRS, add_map_features, add_panel_label

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
FIG_DIR = PAPER_DIR / "figures"

_data_env = os.environ.get("ODEM_DATA")
if not _data_env:
    raise FileNotFoundError(
        "Set ODEM_DATA environment variable to your data directory, e.g.:\n"
        "  export ODEM_DATA=/path/to/dust_model_data"
    )
DATA_ROOT = Path(_data_env)

ERA5_ANNUAL = DATA_ROOT / "output_2006_era5_hourly" / "odem_annual_mean.nc"
M2_ANNUAL   = DATA_ROOT / "output_2006_merra2" / "odem_annual_mean.nc"
DUEM_DIR    = DATA_ROOT / "merra2_data" / "adg"


# ── Load data ─────────────────────────────────────────────────────
def load_odem_annual(path):
    """Load ODEM annual mean emission (F_d_mean in kg m-2 s-1)."""
    ds = nc4.Dataset(str(path))
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    data = ds["F_d_mean"][:]
    ds.close()
    # Mask zeros for log-scale
    data = np.ma.masked_where(data <= 0, data)
    return lat, lon, data


def load_merra2_duem_annual(duem_dir, year=2006):
    """Compute annual mean of MERRA-2 DUEM (sum of 5 bins) from daily files."""
    import glob
    # Filter to target year only
    files = sorted(glob.glob(str(duem_dir / f"*{year}*.nc4")))
    if not files:
        raise FileNotFoundError(f"No MERRA-2 adg files for {year} in {duem_dir}")

    print(f"    Reading {len(files)} MERRA-2 DUEM daily files ({year})...")

    # Read grid from first file
    ds0 = nc4.Dataset(files[0])
    lat = ds0["lat"][:]
    lon = ds0["lon"][:]
    ds0.close()

    # Accumulate daily means
    total = np.zeros((len(lat), len(lon)), dtype=np.float64)
    n_days = 0

    for f in files:
        ds = nc4.Dataset(f)
        # Sum 5 dust size bins, then take daily mean (mean over 24 hourly steps)
        daily_sum = np.zeros((len(lat), len(lon)), dtype=np.float64)
        for i in range(1, 6):
            daily_sum += ds[f"DUEM00{i}"][:].mean(axis=0)
        total += daily_sum
        n_days += 1
        ds.close()

    annual_mean = total / n_days
    annual_mean = np.ma.masked_where(annual_mean <= 0, annual_mean)
    print(f"    {n_days} days averaged")
    return lat, lon, annual_mean


# ── Plot ──────────────────────────────────────────────────────────
def make_figure():
    apply_gmd_style()

    # Shared colorbar limits (log scale)
    vmin, vmax = 1e-12, 1e-6
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = "YlOrBr"  # sequential, warm, color-blind safe

    # Determine which panels are available
    panels = []

    if ERA5_ANNUAL.exists():
        print("  Loading ODEM-ERA5...")
        lat_e, lon_e, data_e = load_odem_annual(ERA5_ANNUAL)
        panels.append(("(a)", "ODEM-ERA5", lat_e, lon_e, data_e))
    else:
        print(f"  SKIP: {ERA5_ANNUAL} not found")

    if M2_ANNUAL.exists():
        print("  Loading ODEM-M2...")
        lat_m, lon_m, data_m = load_odem_annual(M2_ANNUAL)
        panels.append(("(b)", "ODEM-M2", lat_m, lon_m, data_m))
    else:
        print(f"  SKIP: {M2_ANNUAL} not found (run ODEM with MERRA-2 first)")

    if DUEM_DIR.exists():
        print("  Loading MERRA-2 DUEM...")
        lat_d, lon_d, data_d = load_merra2_duem_annual(DUEM_DIR)
        panels.append(("(c)", "MERRA-2 DUEM (online)", lat_d, lon_d, data_d))
    else:
        print(f"  SKIP: {DUEM_DIR} not found")

    if not panels:
        print("  ERROR: No data available for any panel!")
        return

    n = len(panels)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(7.2, 1.5 * n + 0.4),
        subplot_kw={"projection": GMD_PROJ},
    )
    if n == 1:
        axes = [axes]

    for ax, (label, title, lat, lon, data) in zip(axes, panels):
        im = ax.pcolormesh(
            lon, lat, data,
            transform=GMD_DATA_CRS,
            cmap=cmap, norm=norm,
            rasterized=True,
        )
        add_map_features(ax)
        add_panel_label(ax, label)
        ax.set_title(title, fontsize=9, pad=3)

        # Per-panel stats annotation (top-right, away from colorbar)
        total_tg = _flux_to_tg_yr(lat, lon, data)
        ax.text(
            0.98, 0.97,
            f"Global: {total_tg:.0f} Tg yr$^{{-1}}$",
            transform=ax.transAxes, fontsize=7,
            ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85,
                      boxstyle="round,pad=0.3", linewidth=0.4),
        )

    # Shared colorbar
    plt.subplots_adjust(hspace=0.12, top=0.96, bottom=0.10)
    cbar_ax = fig.add_axes([0.18, 0.045, 0.64, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Dust emission flux (kg m$^{-2}$ s$^{-1}$)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f03.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


def _flux_to_tg_yr(lat, lon, flux_kg_m2_s):
    """Convert gridded flux (kg m-2 s-1) to global total (Tg yr-1)."""
    R = 6.371e6  # Earth radius (m)
    dlat = np.abs(np.diff(lat[:2]))[0] * np.pi / 180
    dlon = np.abs(np.diff(lon[:2]))[0] * np.pi / 180

    lat_rad = lat * np.pi / 180
    # Cell area: R² × cos(lat) × dlat × dlon
    area = R**2 * np.abs(np.cos(lat_rad)) * dlat * dlon  # shape: (nlat,)
    area_2d = np.broadcast_to(area[:, np.newaxis], flux_kg_m2_s.shape)

    # Fill masked values with 0 for summation
    flux_filled = np.ma.filled(flux_kg_m2_s, 0.0)
    total_kg_s = np.sum(flux_filled * area_2d)
    total_tg_yr = total_kg_s * 3600 * 24 * 365.25 / 1e9
    return total_tg_yr


if __name__ == "__main__":
    print("ODEM Paper — Figure 3: Annual mean dust emission maps")
    print("=" * 55)
    make_figure()
    print("\nDone!")
