"""
GMD figure style defaults — shared across all ODEM paper figure scripts.

Usage:
    from gmd_style import apply_gmd_style, GMD_PROJ, GMD_DATA_CRS, add_map_features
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def apply_gmd_style():
    """Apply GMD-compliant matplotlib rcParams."""
    plt.rcParams.update({
        # Font: one sans-serif family only (Arial), embedded as TrueType
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Clean look
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    })


# Standard projections
GMD_PROJ = ccrs.Robinson()
GMD_DATA_CRS = ccrs.PlateCarree()


def add_map_features(ax, coastline_width=0.3, border_width=0.15):
    """Add coastlines and borders to a cartopy axes."""
    ax.set_global()
    ax.coastlines(linewidth=coastline_width, color="0.2")
    ax.add_feature(cfeature.BORDERS, linewidth=border_width, edgecolor="0.4")


def add_panel_label(ax, label, fontsize=9):
    """Add (a), (b), etc. panel label to top-left corner."""
    ax.text(
        0.02, 0.97, label,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )
