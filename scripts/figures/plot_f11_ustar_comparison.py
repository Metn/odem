#!/usr/bin/env python3
"""
ODEM Paper — Figure 11: Friction velocity comparison ERA5 vs MERRA-2
=====================================================================
(a) Scatter/density of annual mean u* at source grid cells (ERA5 vs M2)
(b) Zonal mean u* profiles for both reanalyses

Shows how differences in reanalysis u* drive the emission differences.

Output: papers/odem_gmd/figures/f11.pdf + f11.png

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
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmd_style import apply_gmd_style, add_panel_label

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

ERA5_MET_DIR = DATA_ROOT / "era5"
M2_FLX_DIR   = DATA_ROOT / "merra2_data" / "flx"
# Use ODEM annual mean to identify source cells
ERA5_ANNUAL = DATA_ROOT / "output_2006_era5_hourly" / "odem_annual_mean.nc"
M2_ANNUAL   = DATA_ROOT / "output_2006_merra2" / "odem_annual_mean.nc"

YEAR = 2006


def load_era5_ustar_annual(met_dir, year=2006):
    """Compute annual mean ERA5 friction velocity (zust)."""
    files = sorted(glob.glob(str(met_dir / f"era5_dust_{year}*.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 met files in {met_dir}")
    print(f"    Reading {len(files)} ERA5 met files...")

    ds0 = nc4.Dataset(files[0])
    lat = ds0["latitude"][:]
    lon = ds0["longitude"][:]
    ds0.close()

    total = np.zeros((len(lat), len(lon)), dtype=np.float64)
    n = 0
    for f in files:
        ds = nc4.Dataset(f)
        ustar = ds["zust"][:]  # (time, lat, lon)
        total += ustar.mean(axis=0)
        n += 1
        ds.close()
    return lat, lon, total / n


def load_m2_ustar_annual(flx_dir, year=2006):
    """Compute annual mean MERRA-2 friction velocity (USTAR)."""
    files = sorted(glob.glob(str(flx_dir / f"*{year}*.nc4")))
    if not files:
        raise FileNotFoundError(f"No MERRA-2 flx files in {flx_dir}")
    print(f"    Reading {len(files)} MERRA-2 flx files...")

    ds0 = nc4.Dataset(files[0])
    lat = ds0["lat"][:]
    lon = ds0["lon"][:]
    ds0.close()

    total = np.zeros((len(lat), len(lon)), dtype=np.float64)
    n = 0
    for f in files:
        ds = nc4.Dataset(f)
        total += ds["USTAR"][:].mean(axis=0)
        n += 1
        ds.close()
    return lat, lon, total / n


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

    # Load u* data
    print("  Loading ERA5 u*...")
    lat_e, lon_e, ustar_e = load_era5_ustar_annual(ERA5_MET_DIR)
    print("  Loading MERRA-2 u*...")
    lat_m, lon_m, ustar_m = load_m2_ustar_annual(M2_FLX_DIR)

    # Load emission to mask source cells — use M2 grid (coarser)
    print("  Loading emission masks...")
    ds_m = nc4.Dataset(str(M2_ANNUAL))
    em_m2 = ds_m["F_d_mean"][:]
    ds_m.close()

    # Regrid ERA5 u* to M2 grid for scatter comparison
    from scipy.interpolate import RegularGridInterpolator
    # ERA5 lat may be descending
    if lat_e[0] > lat_e[-1]:
        lat_e = lat_e[::-1]
        ustar_e = ustar_e[::-1, :]

    interp = RegularGridInterpolator(
        (lat_e, lon_e), ustar_e,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    lon2d, lat2d = np.meshgrid(lon_m, lat_m)
    ustar_e_regrid = interp((lat2d, lon2d))

    # Source mask: cells where M2 emission > 0
    source_mask = em_m2 > 0
    us_era5 = ustar_e_regrid[source_mask].flatten()
    us_m2 = ustar_m[source_mask].flatten()
    valid = np.isfinite(us_era5) & np.isfinite(us_m2) & (us_era5 > 0) & (us_m2 > 0)
    us_era5 = us_era5[valid]
    us_m2 = us_m2[valid]

    print(f"    {len(us_era5)} source cells for scatter")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})

    # ── Panel (a): 2D density scatter ─────────────────────────
    ax = axes[0]
    xbins = np.linspace(0, 0.8, 100)
    ybins = np.linspace(0, 0.8, 100)
    H, xe, ye = np.histogram2d(us_era5, us_m2, bins=[xbins, ybins])
    H = gaussian_filter(H.T, sigma=1.0)
    H = np.ma.masked_where(H < 0.5, H)

    ax.pcolormesh(xe, ye, H, cmap="YlOrBr", rasterized=True, zorder=1)
    ax.plot([0, 0.8], [0, 0.8], "-", color="0.3", linewidth=0.8, zorder=5)

    # Stats
    r = np.corrcoef(us_era5, us_m2)[0, 1]
    bias = np.mean(us_era5 - us_m2)
    ax.text(0.05, 0.95, f"$R$ = {r:.3f}\nbias = {bias:+.3f} m s$^{{-1}}$",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85,
                      boxstyle="round,pad=0.3", linewidth=0.4))

    ax.set_xlabel(r"ERA5 $u_*$ (m s$^{-1}$)", fontsize=8)
    ax.set_ylabel(r"MERRA-2 $u_*$ (m s$^{-1}$)", fontsize=8)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    _clean_spines(ax)
    add_panel_label(ax, "(a)")

    # ── Panel (b): Zonal mean u* ──────────────────────────────
    ax = axes[1]

    # ERA5 zonal mean (at source cells only, using ERA5 emission)
    ds_e = nc4.Dataset(str(ERA5_ANNUAL))
    em_e = ds_e["F_d_mean"][:]
    lat_e2 = ds_e["lat"][:]
    ds_e.close()

    # ERA5 zonal mean u* — reload at native res
    ustar_e_full = load_era5_ustar_annual.__wrapped__ if hasattr(load_era5_ustar_annual, '__wrapped__') else None
    # Just use already loaded data
    if lat_e[0] > lat_e[-1]:
        # already flipped above
        pass

    # Compute zonal mean at source cells for ERA5
    source_e = em_e > 0
    # Need ERA5 u* at ERA5 grid
    if lat_e2[0] > lat_e2[-1]:
        lat_e2_s = lat_e2[::-1]
        ustar_e_native = ustar_e  # already flipped
        source_e_s = source_e[::-1, :]
    else:
        lat_e2_s = lat_e2
        ustar_e_native = ustar_e
        source_e_s = source_e

    # ERA5 grid may have different lat than met lat — regrid emission mask
    # Actually both are 0.25° so shapes should match
    if ustar_e_native.shape == source_e_s.shape:
        ustar_e_src = np.where(source_e_s, ustar_e_native, np.nan)
        zm_era5 = np.nanmean(ustar_e_src, axis=1)
        lat_zm_e = lat_e
    else:
        # Fallback: just do full zonal mean
        zm_era5 = np.nanmean(ustar_e, axis=1)
        lat_zm_e = lat_e

    # M2 zonal mean at source cells
    ustar_m_src = np.where(source_mask, ustar_m, np.nan)
    zm_m2 = np.nanmean(ustar_m_src, axis=1)

    ax.plot(zm_era5, lat_zm_e, "-", color="#2166ac", linewidth=1.5, label="ERA5")
    ax.plot(zm_m2, lat_m, "-", color="#b2182b", linewidth=1.5, label="MERRA-2")

    ax.set_xlabel(r"$u_*$ (m s$^{-1}$)", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_ylim(-45, 45)
    ax.set_xlim(0, 0.6)
    _clean_spines(ax)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.92, edgecolor="0.75")
    add_panel_label(ax, "(b)")

    # Subtle reference lines
    for yt in [-45, -30, 0, 30, 45]:
        ax.axhline(yt, color="0.88", linewidth=0.3, zorder=0)

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f11.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 11: Friction velocity comparison")
    print("=" * 52)
    make_figure()
    print("\nDone!")
