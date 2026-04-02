#!/usr/bin/env python3
"""
ODEM Paper — Figure 6: Regional dust emission budget comparison
================================================================
Grouped bar chart comparing annual dust emission (Tg yr⁻¹) by region:
  - ODEM-ERA5
  - ODEM-M2
  - MERRA-2 DUEM (online)
  - Kok et al. (2021) observational constraint (with uncertainty)

Regions: North Africa, Middle East, East Asia, North America, South America,
         Southern Africa, Australia, Global.

Output: papers/odem_gmd/figures/f06.pdf + f06.png

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

ERA5_ANNUAL = DATA_ROOT / "output_2006_era5_hourly" / "odem_annual_mean.nc"
M2_ANNUAL   = DATA_ROOT / "output_2006_merra2" / "odem_annual_mean.nc"
DUEM_DIR    = DATA_ROOT / "merra2_data" / "adg"


# ── Region definitions (lat/lon bounding boxes) ──────────────────
REGIONS = {
    "N. Africa":     {"lat": (0, 40),    "lon": (-20, 40)},
    "Middle East":   {"lat": (12, 45),   "lon": (40, 65)},
    "East Asia":     {"lat": (25, 50),   "lon": (70, 130)},
    "N. America":    {"lat": (15, 50),   "lon": (-120, -80)},
    "S. America":    {"lat": (-56, 0),   "lon": (-80, -35)},
    "S. Africa":     {"lat": (-35, 0),   "lon": (10, 50)},
    "Australia":     {"lat": (-45, -10), "lon": (110, 155)},
}

# Kok et al. (2021) Table 1 — PM20 values (direct, no conversion)
# mean, low, high (Tg yr-1, PM20)
KOK2021 = {
    "N. Africa":     (2727, 730, 11000),
    "Middle East":   (398,  100,  1700),
    "East Asia":     (500,  130,  2000),
    "N. America":    (50,   10,   300),
    "S. America":    (90,   20,   500),
    "S. Africa":     (70,   15,   350),
    "Australia":     (180,  40,   900),
    "Global":        (5000, 3400, 6600),
}


def compute_regional_budget(lat, lon, flux):
    """Compute regional emission totals (Tg yr-1) from gridded flux."""
    R = 6.371e6
    dlat = np.abs(np.diff(lat[:2]))[0] * np.pi / 180
    dlon = np.abs(np.diff(lon[:2]))[0] * np.pi / 180
    lat_rad = lat * np.pi / 180
    area = R**2 * np.abs(np.cos(lat_rad)) * dlat * dlon

    flux_filled = np.ma.filled(flux, 0.0)
    sec_per_yr = 3600 * 24 * 365.25

    budgets = {}
    for name, box in REGIONS.items():
        lat_mask = (lat >= box["lat"][0]) & (lat <= box["lat"][1])
        lon_mask = (lon >= box["lon"][0]) & (lon <= box["lon"][1])
        region_flux = flux_filled[np.ix_(lat_mask, lon_mask)]
        region_area = area[lat_mask]
        region_area_2d = np.broadcast_to(region_area[:, np.newaxis], region_flux.shape)
        budgets[name] = np.sum(region_flux * region_area_2d) * sec_per_yr / 1e9

    # Global
    area_2d = np.broadcast_to(area[:, np.newaxis], flux_filled.shape)
    budgets["Global"] = np.sum(flux_filled * area_2d) * sec_per_yr / 1e9

    return budgets


def load_odem_annual(path):
    """Load ODEM annual mean emission."""
    ds = nc4.Dataset(str(path))
    lat, lon = ds["lat"][:], ds["lon"][:]
    data = ds["F_d_mean"][:]
    ds.close()
    return lat, lon, np.ma.masked_where(data <= 0, data)


def load_merra2_duem_annual(duem_dir, year=2006):
    """Compute annual mean MERRA-2 DUEM from daily files."""
    import glob
    files = sorted(glob.glob(str(duem_dir / f"*{year}*.nc4")))
    if not files:
        return None, None, None

    ds0 = nc4.Dataset(files[0])
    lat, lon = ds0["lat"][:], ds0["lon"][:]
    ds0.close()

    total = np.zeros((len(lat), len(lon)), dtype=np.float64)
    for f in files:
        ds = nc4.Dataset(f)
        for i in range(1, 6):
            total += ds[f"DUEM00{i}"][:].mean(axis=0)
        ds.close()

    annual_mean = total / len(files)
    return lat, lon, np.ma.masked_where(annual_mean <= 0, annual_mean)


def make_figure():
    apply_gmd_style()

    # Load available datasets
    datasets = {}

    if ERA5_ANNUAL.exists():
        print("  Loading ODEM-ERA5...")
        lat, lon, flux = load_odem_annual(ERA5_ANNUAL)
        datasets["ODEM-ERA5"] = compute_regional_budget(lat, lon, flux)

    if M2_ANNUAL.exists():
        print("  Loading ODEM-M2...")
        lat, lon, flux = load_odem_annual(M2_ANNUAL)
        datasets["ODEM-M2"] = compute_regional_budget(lat, lon, flux)

    if DUEM_DIR.exists():
        print("  Loading MERRA-2 DUEM...")
        lat, lon, flux = load_merra2_duem_annual(DUEM_DIR)
        if lat is not None:
            datasets["MERRA-2 DUEM"] = compute_regional_budget(lat, lon, flux)

    if not datasets:
        print("  ERROR: No data available!")
        return

    # Print budget table
    regions = list(REGIONS.keys()) + ["Global"]
    print(f"\n  {'Region':<14s}", end="")
    for name in datasets:
        print(f"  {name:>14s}", end="")
    print(f"  {'Kok2021':>14s}")
    for reg in regions:
        print(f"  {reg:<14s}", end="")
        for name, bud in datasets.items():
            print(f"  {bud[reg]:>14.0f}", end="")
        if reg in KOK2021:
            print(f"  {KOK2021[reg][0]:>14.0f}", end="")
        print()

    # ── Plot ──────────────────────────────────────────────────
    colors = {
        "ODEM-ERA5":    "#2166ac",  # blue
        "ODEM-M2":      "#b2182b",  # red
        "MERRA-2 DUEM": "#d4a017",  # amber (visible on white)
    }
    kok_color = "#4dac26"  # green

    n_regions = len(regions)
    n_datasets = len(datasets)
    bar_width = 0.22
    x = np.arange(n_regions)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    # Dataset bars
    for i, (name, bud) in enumerate(datasets.items()):
        vals = [bud[r] for r in regions]
        offset = (i - n_datasets / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width * 0.88,
               label=name, color=colors.get(name, f"C{i}"),
               edgecolor="white", linewidth=0.6, zorder=3)

    # Kok 2021 markers with error bars
    kok_means = [KOK2021[r][0] for r in regions]
    kok_lo = [KOK2021[r][0] - KOK2021[r][1] for r in regions]
    kok_hi = [KOK2021[r][2] - KOK2021[r][0] for r in regions]
    offset_kok = (n_datasets / 2 + 0.5) * bar_width
    ax.errorbar(x + offset_kok, kok_means, yerr=[kok_lo, kok_hi],
                fmt="D", markersize=4.5, color=kok_color,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=3, capthick=0.8, elinewidth=0.8,
                label="Kok et al. (2021)", zorder=5)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("0.3")
    ax.tick_params(width=0.6, colors="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel(r"Dust emission (Tg yr$^{-1}$)", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(10, 50000)

    # Subtle horizontal reference lines
    for yt in [10, 100, 1000, 10000]:
        ax.axhline(yt, color="0.88", linewidth=0.3, zorder=0)

    # Legend — inside, upper center
    leg = ax.legend(fontsize=6.5, loc="upper center", ncol=4,
                    framealpha=0.95, edgecolor="0.8",
                    borderpad=0.4, handlelength=1.5,
                    columnspacing=1.0)
    leg.get_frame().set_linewidth(0.4)

    plt.tight_layout()

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f06.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 6: Regional dust emission budget")
    print("=" * 55)
    make_figure()
    print("\nDone!")
