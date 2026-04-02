#!/usr/bin/env python3
"""
ODEM Paper — Figure 10: Emission flux vs friction velocity relationship
========================================================================
Theoretical Kok (2014) emission curve vs ODEM-simulated emission,
sampled from the diagnostic output.

Shows that the model correctly implements the brittle fragmentation
emission equation: F_d ∝ (u*² - u*_t²) with fragmentation exponent.

  (a) Theoretical curve + ODEM scatter (grid-cell monthly means)
  (b) Histogram of u* distribution at source regions

Output: papers/odem_gmd/figures/f10.pdf + f10.png

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
FIG_DIR = Path(os.environ.get("ODEM_FIG_DIR", "./figures"))

_data_env = os.environ.get("ODEM_DATA")
if not _data_env:
    raise FileNotFoundError(
        "Set ODEM_DATA environment variable to your data directory, e.g.:
"
        "  export ODEM_DATA=/path/to/dust_model_data"
    )
DATA_ROOT = Path(_data_env)

DIAG_FILE = DATA_ROOT / "output_2006_diag" / "odem_diag_mean.nc"
EMISSION_FILE = DATA_ROOT / "output_2006_diag" / "odem_200601.nc"


# ── Kok (2014) theoretical curve ─────────────────────────────────
def kok2014_emission(ustar_s, ustar_t, D_p=127e-6):
    """Kok (2014) brittle fragmentation emission flux (kg m-2 s-1).

    Simplified version with default parameters for illustration.
    """
    C_tune = 0.05
    C_d0 = 4.4e-5
    C_e = 2.0
    C_alpha = 2.7
    ustar_st0 = 0.16
    rho_a = 1.225  # standard air density

    # Standardized threshold
    ustar_st_salt = ustar_t * np.sqrt(rho_a / 1.225)

    # Dust emission coefficient
    C_d = C_d0 * np.exp(-C_e * (ustar_st_salt - ustar_st0) / ustar_st0)

    # Fragmentation exponent (capped at 3)
    alpha = np.minimum(C_alpha * ustar_st_salt / ustar_s, 3.0)

    # Emission
    F = C_tune * C_d * (ustar_s**2 - ustar_t**2) / ustar_st_salt
    F *= (ustar_s / ustar_t) ** alpha

    # Zero below threshold
    F = np.where(ustar_s > ustar_t, F, 0.0)
    return F


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
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter
    from gmd_style import add_panel_label

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.32})

    # ── Panel (a): Theoretical curve + ODEM density ─────────
    ax = axes[0]

    # Load ODEM data first (need it for density plot behind theory curves)
    has_odem = False
    if DIAG_FILE.exists() and EMISSION_FILE.exists():
        print("  Loading ODEM diagnostics...")
        ds_d = nc4.Dataset(str(DIAG_FILE))
        ds_e = nc4.Dataset(str(EMISSION_FILE))

        ustar_s_data = ds_d["wnd_frc_slt"][:] if "wnd_frc_slt" in ds_d.variables else None
        ustar_t_data = ds_d["ustar_ft_wet"][:] if "ustar_ft_wet" in ds_d.variables else None

        if "F_d" in ds_e.variables:
            emission = ds_e["F_d"][:].mean(axis=0)
        elif "F_d_mean" in ds_e.variables:
            emission = ds_e["F_d_mean"][:]
        else:
            emission = None

        if ustar_s_data is not None and ustar_t_data is not None and emission is not None:
            mask = (emission > 0) & (ustar_s_data > 0) & (ustar_t_data > 0)
            us_flat = np.array(ustar_s_data[mask]).flatten()
            em_flat = np.array(emission[mask]).flatten()

            # 2D histogram as density heatmap (much better than scatter blob)
            log_em = np.log10(em_flat)
            xbins = np.linspace(0, 1.2, 120)
            ybins = np.linspace(-14, -5, 100)
            H, xedges, yedges = np.histogram2d(us_flat, log_em,
                                                bins=[xbins, ybins])
            H = gaussian_filter(H.T, sigma=1.2)
            H = np.ma.masked_where(H < 0.5, H)

            ax.pcolormesh(xedges, yedges, H,
                          cmap="Greys", alpha=0.7, rasterized=True, zorder=1)
            # Contour overlay for structure
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            levels = np.percentile(H.compressed(), [50, 75, 90, 97])
            ax.contour(xc, yc, H, levels=levels,
                       colors="0.4", linewidths=0.4, alpha=0.5, zorder=2)
            has_odem = True

        ds_d.close()
        ds_e.close()
    else:
        print("  SKIP: Diagnostic files not found, plotting theory only")

    # Theoretical curves — on top of density
    ustar_range = np.linspace(0.01, 1.5, 500)
    curve_styles = [
        (0.20, "#2166ac", r"$u^*_t = 0.20$ m s$^{-1}$", "-"),
        (0.30, "#b2182b", r"$u^*_t = 0.30$ m s$^{-1}$", "--"),
        (0.40, "#4dac26", r"$u^*_t = 0.40$ m s$^{-1}$", "-."),
    ]
    for ustar_t, color, label, ls in curve_styles:
        F = kok2014_emission(ustar_range, ustar_t)
        F[F <= 0] = np.nan
        log_F = np.log10(F)
        ax.plot(ustar_range, log_F, ls, color=color, linewidth=1.8,
                label=label, zorder=5, alpha=0.9)

    if has_odem:
        # Ghost handle for ODEM density in legend
        from matplotlib.patches import Patch
        odem_patch = Patch(facecolor="0.65", alpha=0.5, label="ODEM (Jan 2006)")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(odem_patch)
        labels.append("ODEM (Jan 2006)")
        ax.legend(handles, labels, fontsize=6, loc="lower right",
                  framealpha=0.92, edgecolor="0.75", borderpad=0.5)
    else:
        ax.legend(fontsize=6, loc="lower right",
                  framealpha=0.92, edgecolor="0.75")

    ax.set_xlabel(r"Surface friction velocity $u^*_s$ (m s$^{-1}$)", fontsize=8)
    ax.set_ylabel(r"Emission flux $\log_{10}\,F_d$ (kg m$^{-2}$ s$^{-1}$)", fontsize=8)
    ax.set_ylim(-14, -5)
    ax.set_xlim(0, 1.2)
    _clean_spines(ax)
    # Subtle reference lines
    for yt in range(-13, -5):
        ax.axhline(yt, color="0.88", linewidth=0.3, zorder=0)
    add_panel_label(ax, "(a)")

    # ── Panel (b): u* distribution — KDE + rug ──────────────
    ax = axes[1]

    if DIAG_FILE.exists():
        ds_d = nc4.Dataset(str(DIAG_FILE))
        lat = ds_d["lat"][:] if "lat" in ds_d.variables else None
        lon = ds_d["lon"][:] if "lon" in ds_d.variables else None
        ustar_s_data = ds_d["wnd_frc_slt"][:] if "wnd_frc_slt" in ds_d.variables else None
        ds_d.close()

        if ustar_s_data is not None and lat is not None:
            regions = {
                "N. Africa": {"lat": (15, 35), "lon": (-10, 35)},
                "Arabia":    {"lat": (15, 35), "lon": (35, 60)},
                "E. Asia":   {"lat": (30, 45), "lon": (75, 115)},
            }
            colors = ["#2166ac", "#b2182b", "#4dac26"]

            for (name, box), color in zip(regions.items(), colors):
                lat_mask = (lat >= box["lat"][0]) & (lat <= box["lat"][1])
                lon_mask = (lon >= box["lon"][0]) & (lon <= box["lon"][1])
                region_ustar = ustar_s_data[np.ix_(lat_mask, lon_mask)].flatten()
                region_ustar = region_ustar[region_ustar > 0.01]
                if len(region_ustar) == 0:
                    continue

                # KDE via scipy
                from scipy.stats import gaussian_kde
                xgrid = np.linspace(0, 1.2, 300)
                kde = gaussian_kde(region_ustar, bw_method=0.08)
                density = kde(xgrid)

                # Filled KDE curve
                ax.fill_between(xgrid, density, alpha=0.15, color=color, zorder=2)
                ax.plot(xgrid, density, "-", color=color, linewidth=1.5,
                        label=name, zorder=3)

                # Median vertical line
                med = np.median(region_ustar)
                ymax = kde(med)[0]
                ax.vlines(med, 0, ymax, color=color, linewidth=0.8,
                          linestyle=":", alpha=0.7, zorder=4)
                ax.plot(med, ymax, "v", color=color, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)

            ax.set_xlabel(r"$u^*_s$ (m s$^{-1}$)", fontsize=8)
            ax.set_ylabel("Probability density", fontsize=8)
            ax.set_xlim(0, 1.0)
            ax.set_ylim(bottom=0)
            leg = ax.legend(fontsize=6.5, loc="upper right",
                            framealpha=0.92, edgecolor="0.75", borderpad=0.5,
                            title="median " + r"$\blacktriangledown$",
                            title_fontsize=5.5)
            _clean_spines(ax)
            # Subtle reference lines
            for yt in ax.get_yticks():
                if yt > 0:
                    ax.axhline(yt, color="0.88", linewidth=0.3, zorder=0)
    else:
        ax.text(0.5, 0.5, "Diagnostics not\navailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.5")

    # Panel label — mid-right, below legend, above data
    ax.text(0.45, 0.85, "(b)", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"f10.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    print("ODEM Paper — Figure 10: Emission flux vs u* relationship")
    print("=" * 58)
    make_figure()
    print("\nDone!")
