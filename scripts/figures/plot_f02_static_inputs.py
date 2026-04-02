#!/usr/bin/env python3
"""
ODEM Paper — Figure 2: Static input datasets
===============================================
Four-panel map showing the key static inputs to ODEM v1.0:
  (a) Dust source erodibility mask (MODIS land cover)
  (b) Topsoil clay fraction (SoilGrids v2.0)
  (c) Aerodynamic roughness length z0a (Prigent et al., 2005)
  (d) Annual mean leaf area index (MODIS LAI, 2006)

Output: papers/odem_gmd/figures/f02.pdf + f02.png

GMD compliance:
  - Sans-serif font (Arial/Helvetica)
  - Panel labels (a), (b), (c), (d)
  - Color-blind safe colormaps
  - 300 DPI minimum
  - Full-width figure (figure*)

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Paths (adjust for desktop / laptop) ──────────────────────────
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

EROD_FILE = DATA_ROOT / "modis_landcover" / "modis_landcover_erodibility_2006.nc"
CLAY_FILE = DATA_ROOT / "soilgrids" / "SG_clay.nc4"
Z0A_FILE  = DATA_ROOT / "prigent_data" / "Prigent_2005_roughness_0.25x0.25.nc"
LAI_DIR   = DATA_ROOT / "modis_lai" / "2006"


# ── Load data ────────────────────────────────────────────────────
def load_erodibility():
    """Load dust source erodibility mask (0–1)."""
    ds = nc4.Dataset(str(EROD_FILE))
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    data = ds["f_erodible"][:]
    ds.close()
    # Mask zeros for cleaner plotting
    data = np.ma.masked_where(data == 0, data)
    return lat, lon, data


def load_clay():
    """Load topsoil clay fraction (g/kg → %)."""
    ds = nc4.Dataset(str(CLAY_FILE))
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    data = ds["T_CLAY"][:]  # already in %
    ds.close()
    data = np.ma.masked_invalid(data)
    return lat, lon, data


def load_z0a():
    """Load Prigent aerodynamic roughness z0a (m), log-scale."""
    ds = nc4.Dataset(str(Z0A_FILE))
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    data = ds["Z0a"][0, :, :]
    ds.close()
    data = np.ma.masked_where(data <= 0, data)
    return lat, lon, data


def load_lai_annual_mean():
    """Load MODIS LAI annual mean (all 8-day composites in 2006)."""
    files = sorted(LAI_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No LAI files in {LAI_DIR}")

    # Read first file for coordinates
    ds0 = nc4.Dataset(str(files[0]))
    lat = ds0["lat"][:]
    lon = ds0["lon"][:]
    ds0.close()

    # Stack all timesteps and compute mean
    lai_stack = []
    for f in files:
        ds = nc4.Dataset(str(f))
        lai = ds["lai"][0, :, :]  # shape: (lat, lon)
        lai_stack.append(lai)
        ds.close()

    lai_mean = np.ma.mean(np.ma.stack(lai_stack), axis=0)
    return lat, lon, lai_mean


# ── Plot ─────────────────────────────────────────────────────────
def make_figure():
    # GMD: one sans-serif family only (Arial), TrueType embedding, math in Arial
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,       # TrueType outlines (not Type 3 bitmaps)
        "ps.fonttype": 42,
    })

    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.2, 4.8),  # ~18 cm wide (GMD full-width)
        subplot_kw={"projection": proj},
    )

    # ── Panel definitions ─────────────────────────────────────
    panels = [
        {
            "label": "(a)",
            "title": "Dust source erodibility",
            "loader": load_erodibility,
            "cmap": "YlOrBr",
            "norm": None,
            "vmin": 0, "vmax": 1,
            "cbar_label": "Erodibility fraction",
            "log": False,
        },
        {
            "label": "(b)",
            "title": "Topsoil clay fraction (SoilGrids)",
            "loader": load_clay,
            "cmap": "YlOrRd",
            "norm": None,
            "vmin": 0, "vmax": 60,
            "cbar_label": "Clay (%)",
            "log": False,
        },
        {
            "label": "(c)",
            "title": r"Aerodynamic roughness $z_{0a}$ (Prigent)",
            "loader": load_z0a,
            "cmap": "viridis",
            "norm": mcolors.LogNorm(vmin=1e-4, vmax=1.0),
            "vmin": None, "vmax": None,
            "cbar_label": r"$z_{0a}$ (m)",
            "log": True,
        },
        {
            "label": "(d)",
            "title": "Annual mean LAI (MODIS, 2006)",
            "loader": load_lai_annual_mean,
            "cmap": "YlGn",
            "norm": None,
            "vmin": 0, "vmax": 5,
            "cbar_label": r"LAI (m$^2$ m$^{-2}$)",
            "log": False,
        },
    ]

    for ax, panel in zip(axes.flat, panels):
        print(f"  Loading {panel['title']}...")
        lat, lon, data = panel["loader"]()

        # Plot
        kwargs = dict(
            transform=data_crs,
            cmap=panel["cmap"],
        )
        if panel["norm"] is not None:
            kwargs["norm"] = panel["norm"]
        else:
            kwargs["vmin"] = panel["vmin"]
            kwargs["vmax"] = panel["vmax"]

        im = ax.pcolormesh(lon, lat, data, **kwargs, rasterized=True)

        # Map features
        ax.set_global()
        ax.coastlines(linewidth=0.3, color="0.2")
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, edgecolor="0.4")

        # Panel label — top-left, bold
        ax.text(
            0.02, 0.97, panel["label"],
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

        # Title below panel label
        ax.set_title(panel["title"], fontsize=8, pad=3)

        # Colorbar
        cbar = fig.colorbar(
            im, ax=ax, orientation="horizontal",
            fraction=0.06, pad=0.04, aspect=25,
            shrink=0.85,
        )
        cbar.set_label(panel["cbar_label"], fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    plt.subplots_adjust(wspace=0.08, hspace=0.22, left=0.02, right=0.98,
                        top=0.95, bottom=0.05)

    # ── Save ──────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / "f02.pdf"
    png_path = FIG_DIR / "f02.png"

    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    print(f"  PDF size: {pdf_path.stat().st_size / 1024:.0f} KB")
    print(f"  PNG size: {png_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    print("ODEM Paper — Figure 2: Static input datasets")
    print("=" * 50)
    make_figure()
    print("\nDone!")
