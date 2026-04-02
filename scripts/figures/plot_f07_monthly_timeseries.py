#!/usr/bin/env python3
"""
ODEM Paper — Figure 7: Monthly global dust emission timeseries
===============================================================
Line plot of monthly global emission totals (Tg/month) for 2006:
  - ODEM-ERA5
  - ODEM-M2
  - MERRA-2 DUEM (online)
  - Kok et al. (2021) annual mean ± uncertainty (shaded band)

Output: papers/odem_gmd/figures/f07.pdf + f07.png

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
from calendar import monthrange

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmd_style import apply_gmd_style

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

ERA5_DIR  = DATA_ROOT / "output_2006_era5_hourly"
M2_DIR    = DATA_ROOT / "output_2006_merra2"
DUEM_DIR  = DATA_ROOT / "merra2_data" / "adg"

YEAR = 2006
MONTHS = range(1, 13)
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def flux_to_tg_month(lat, lon, flux, year, month):
    """Convert gridded flux (kg m-2 s-1) to monthly total (Tg)."""
    R = 6.371e6
    dlat = np.abs(np.diff(lat[:2]))[0] * np.pi / 180
    dlon = np.abs(np.diff(lon[:2]))[0] * np.pi / 180
    lat_rad = lat * np.pi / 180
    area = R**2 * np.abs(np.cos(lat_rad)) * dlat * dlon
    area_2d = np.broadcast_to(area[:, np.newaxis], flux.shape)

    days_in_month = monthrange(year, month)[1]
    sec_in_month = days_in_month * 24 * 3600

    flux_filled = np.ma.filled(flux, 0.0)
    return np.sum(flux_filled * area_2d) * sec_in_month / 1e9


def load_odem_monthly(output_dir):
    """Load ODEM monthly files and compute monthly global totals."""
    monthly_tg = []
    for m in MONTHS:
        fname = output_dir / f"odem_{YEAR}{m:02d}.nc"
        if not fname.exists():
            monthly_tg.append(np.nan)
            continue
        ds = nc4.Dataset(str(fname))
        lat, lon = ds["lat"][:], ds["lon"][:]
        # Monthly files contain timestep data — compute time-mean first
        if "F_d" in ds.variables:
            flux = ds["F_d"][:].mean(axis=0)  # mean over timesteps
        elif "F_d_mean" in ds.variables:
            flux = ds["F_d_mean"][:]
        else:
            # Try to find emission variable
            for vname in ds.variables:
                if "emission" in (getattr(ds[vname], "long_name", "") or "").lower():
                    flux = ds[vname][:].mean(axis=0)
                    break
            else:
                print(f"    WARNING: no emission var in {fname}")
                monthly_tg.append(np.nan)
                ds.close()
                continue
        ds.close()
        monthly_tg.append(flux_to_tg_month(lat, lon, flux, YEAR, m))
    return np.array(monthly_tg)


def load_merra2_duem_monthly(duem_dir):
    """Compute monthly MERRA-2 DUEM totals from daily files."""
    import glob

    monthly_tg = []
    for m in MONTHS:
        pattern = str(duem_dir / f"*{YEAR}{m:02d}*.nc4")
        files = sorted(glob.glob(pattern))
        if not files:
            monthly_tg.append(np.nan)
            continue

        # Get grid
        ds0 = nc4.Dataset(files[0])
        lat, lon = ds0["lat"][:], ds0["lon"][:]
        ds0.close()

        # Average daily means for the month
        total = np.zeros((len(lat), len(lon)), dtype=np.float64)
        for f in files:
            ds = nc4.Dataset(f)
            for i in range(1, 6):
                total += ds[f"DUEM00{i}"][:].mean(axis=0)
            ds.close()
        monthly_mean_flux = total / len(files)

        monthly_tg.append(flux_to_tg_month(lat, lon, monthly_mean_flux, YEAR, m))

    return np.array(monthly_tg)


def _clean_spines(ax):
    """Remove top/right spines, soften remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("0.3")
    ax.tick_params(width=0.6, colors="0.3", labelsize=7)


def make_figure():
    apply_gmd_style()

    x = np.arange(12)

    # ── Load data ─────────────────────────────────────────────
    has_data = False
    curves = []

    if ERA5_DIR.exists() and (ERA5_DIR / f"odem_{YEAR}01.nc").exists():
        print("  Loading ODEM-ERA5 monthly...")
        era5_tg = load_odem_monthly(ERA5_DIR)
        curves.append(("ODEM-ERA5", era5_tg, "#2166ac", "o"))
        has_data = True
        print(f"    Annual total: {np.nansum(era5_tg):.0f} Tg")

    if M2_DIR.exists() and (M2_DIR / f"odem_{YEAR}01.nc").exists():
        print("  Loading ODEM-M2 monthly...")
        m2_tg = load_odem_monthly(M2_DIR)
        curves.append(("ODEM-M2", m2_tg, "#b2182b", "s"))
        has_data = True
        print(f"    Annual total: {np.nansum(m2_tg):.0f} Tg")

    if DUEM_DIR.exists():
        print("  Loading MERRA-2 DUEM monthly...")
        duem_tg = load_merra2_duem_monthly(DUEM_DIR)
        curves.append(("MERRA-2 DUEM", duem_tg, "#d4a017", "^"))
        has_data = True
        print(f"    Annual total: {np.nansum(duem_tg):.0f} Tg")

    if not has_data:
        print("  ERROR: No data available!")
        return

    # ── Single clean panel, linear scale ──────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    # Kok (2021) PM20 band — fill_between on x=0..11 so it stays within data range
    kok_mean_monthly = 5000 / 12
    kok_lo_monthly = 3400 / 12
    kok_hi_monthly = 6600 / 12
    ax.fill_between(x, kok_lo_monthly, kok_hi_monthly,
                    color="#4dac26", alpha=0.12, zorder=1,
                    label="Kok et al. (2021) range")
    ax.plot(x, [kok_mean_monthly] * 12, color="#4dac26", linewidth=0.7,
            linestyle=":", alpha=0.5, zorder=1)

    # Plot curves — lines + markers only, no fill
    for name, data, color, marker in curves:
        valid = ~np.isnan(data)
        ax.plot(x[valid], data[valid], "-", color=color,
                linewidth=1.8, zorder=3)
        ax.plot(x[valid], data[valid], marker, color=color,
                markersize=5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4, label=name)

    # Axis styling
    _clean_spines(ax)
    ax.set_xlim(0, 11)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS, fontsize=7)
    ax.set_ylabel(r"Dust emission (Tg month$^{-1}$)", fontsize=8)
    ax.set_xlabel("2006", fontsize=8)

    # Subtle horizontal reference lines
    for yt in ax.get_yticks():
        if yt > 0:
            ax.axhline(yt, color="0.88", linewidth=0.3, zorder=0)

    # Legend — inside plot, top right
    leg = ax.legend(fontsize=6.5, loc="upper right", ncol=2,
                    framealpha=0.95, edgecolor="0.8",
                    borderpad=0.4, handlelength=1.8,
                    columnspacing=1.2)
    leg.get_frame().set_linewidth(0.4)

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f07.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 7: Monthly global dust emission timeseries")
    print("=" * 60)
    make_figure()
    print("\nDone!")
