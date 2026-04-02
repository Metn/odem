#!/usr/bin/env python3
"""
ODEM Paper — Figure 9: ODEM emission vs DustCOMM dust loading
================================================================
Two-row comparison:
  (a) ODEM-ERA5 annual emission  (b) ODEM-M2 annual emission
  (c) DustCOMM annual dust loading (mean)
  (d) Scatter: ODEM emission vs DustCOMM loading at matched grid cells

Shows that ODEM source regions are consistent with the DustCOMM
observationally constrained dust loading pattern.

Output: papers/odem_gmd/figures/f09.pdf + f09.png

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

ERA5_ANNUAL  = DATA_ROOT / "output_2006_era5_hourly" / "odem_annual_mean.nc"
M2_ANNUAL    = DATA_ROOT / "output_2006_merra2" / "odem_annual_mean.nc"
DUSTCOMM_FILE = DATA_ROOT / "dustcomm" / "Dust_Load_annual.nc"


def _clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("0.3")
    ax.tick_params(width=0.6, colors="0.3", labelsize=7)


def make_figure():
    apply_gmd_style()

    # ── Load ODEM annual means ────────────────────────────────
    print("  Loading ODEM-ERA5...")
    ds_e = nc4.Dataset(str(ERA5_ANNUAL))
    lat_e, lon_e = ds_e["lat"][:], ds_e["lon"][:]
    em_e = ds_e["F_d_mean"][:]
    ds_e.close()
    em_e = np.ma.masked_where(em_e <= 0, em_e)

    print("  Loading ODEM-M2...")
    ds_m = nc4.Dataset(str(M2_ANNUAL))
    lat_m, lon_m = ds_m["lat"][:], ds_m["lon"][:]
    em_m = ds_m["F_d_mean"][:]
    ds_m.close()
    em_m = np.ma.masked_where(em_m <= 0, em_m)

    # ── Load DustCOMM ─────────────────────────────────────────
    print("  Loading DustCOMM...")
    ds_dc = nc4.Dataset(str(DUSTCOMM_FILE))
    lat_dc = ds_dc["lat"][:]
    lon_dc = ds_dc["lon"][:]
    # Note: DustCOMM is stored as (lon, lat), need to transpose
    load_dc = ds_dc["Latm_mean"][:].T  # now (lat, lon)
    ds_dc.close()
    load_dc = np.ma.masked_where(load_dc <= 0, load_dc)
    print(f"    DustCOMM shape: {load_dc.shape}, range: {load_dc.min():.4f}–{load_dc.max():.2f}")

    # ── Figure: 2×2 layout ────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 6.5))

    # Top row: emission maps
    ax1 = fig.add_subplot(2, 2, 1, projection=GMD_PROJ)
    ax2 = fig.add_subplot(2, 2, 2, projection=GMD_PROJ)
    # Bottom left: DustCOMM map
    ax3 = fig.add_subplot(2, 2, 3, projection=GMD_PROJ)
    # Bottom right: scatter
    ax4 = fig.add_subplot(2, 2, 4)

    em_norm = mcolors.LogNorm(vmin=1e-12, vmax=1e-6)
    em_cmap = "YlOrBr"

    # (a) ODEM-ERA5
    im1 = ax1.pcolormesh(lon_e, lat_e, em_e, transform=GMD_DATA_CRS,
                          cmap=em_cmap, norm=em_norm, rasterized=True)
    add_map_features(ax1)
    add_panel_label(ax1, "(a)")
    ax1.set_title("ODEM-ERA5 emission", fontsize=8, pad=3)

    # (b) ODEM-M2
    im2 = ax2.pcolormesh(lon_m, lat_m, em_m, transform=GMD_DATA_CRS,
                          cmap=em_cmap, norm=em_norm, rasterized=True)
    add_map_features(ax2)
    add_panel_label(ax2, "(b)")
    ax2.set_title("ODEM-M2 emission", fontsize=8, pad=3)

    # (c) DustCOMM loading
    dc_norm = mcolors.LogNorm(vmin=0.001, vmax=3.0)
    im3 = ax3.pcolormesh(lon_dc, lat_dc, load_dc, transform=GMD_DATA_CRS,
                          cmap="YlGn", norm=dc_norm, rasterized=True)
    add_map_features(ax3)
    add_panel_label(ax3, "(c)")
    ax3.set_title("DustCOMM dust loading", fontsize=8, pad=3)

    # Colorbars for maps — placed dynamically from axes positions
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    fig.canvas.draw()

    p1 = ax1.get_position()
    p2 = ax2.get_position()
    p3 = ax3.get_position()

    # Emission colorbar: centred under top row, gap below ax1/ax2 bottom
    cbar1_ax = fig.add_axes([p1.x0, p1.y0 - 0.055, p2.x1 - p1.x0, 0.015])
    cbar1 = fig.colorbar(im1, cax=cbar1_ax, orientation="horizontal")
    cbar1.set_label(r"Emission (kg m$^{-2}$ s$^{-1}$)", fontsize=6.5)
    cbar1.ax.tick_params(labelsize=6)

    # DustCOMM colorbar: below ax3
    cbar3_ax = fig.add_axes([p3.x0, p3.y0 - 0.055, p3.width, 0.015])
    cbar3 = fig.colorbar(im3, cax=cbar3_ax, orientation="horizontal")
    cbar3.set_label("Dust loading (g m$^{-2}$)", fontsize=6.5)
    cbar3.ax.tick_params(labelsize=6)

    # ── (d) Scatter: ODEM emission vs DustCOMM loading ────────
    # Regrid ODEM-M2 emission to DustCOMM grid
    from scipy.interpolate import RegularGridInterpolator

    lat_m_arr = np.array(lat_m)
    lon_m_arr = np.array(lon_m)
    em_m_filled = np.ma.filled(em_m, 0.0)

    if lat_m_arr[0] > lat_m_arr[-1]:
        lat_m_arr = lat_m_arr[::-1]
        em_m_filled = em_m_filled[::-1, :]

    interp = RegularGridInterpolator(
        (lat_m_arr, lon_m_arr), em_m_filled,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    lon2d_dc, lat2d_dc = np.meshgrid(lon_dc, lat_dc)
    em_on_dc = interp((lat2d_dc, lon2d_dc))

    # Only where both have signal
    min_em = 1e-13
    min_load = 0.001
    valid = (em_on_dc > min_em) & (np.array(load_dc) > min_load)
    x = em_on_dc[valid].flatten()
    y = np.array(load_dc)[valid].flatten()

    ax4.scatter(x, y, s=4, alpha=0.3, color="#b2182b", edgecolors="none",
                rasterized=True, zorder=2)

    # Correlation
    r = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
    ax4.text(0.95, 0.05, f"$R$ (log-log) = {r:.2f}\n$N$ = {len(x)}",
             transform=ax4.transAxes, fontsize=7, va="bottom", ha="right",
             bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85,
                       boxstyle="round,pad=0.3", linewidth=0.4))

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel(r"ODEM-M2 emission (kg m$^{-2}$ s$^{-1}$)", fontsize=7)
    ax4.set_ylabel("DustCOMM loading (g m$^{-2}$)", fontsize=7, labelpad=-1)
    ax4.set_xlim(1e-13, 1e-6)
    ax4.set_ylim(0.001, 5)
    _clean_spines(ax4)
    add_panel_label(ax4, "(d)")

    # Subtle ref lines
    for yt in [0.01, 0.1, 1.0]:
        ax4.axhline(yt, color="0.88", linewidth=0.3, zorder=0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f09.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 9: ODEM vs DustCOMM comparison")
    print("=" * 52)
    make_figure()
    print("\nDone!")
