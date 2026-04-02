#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

"""
ODEM — Analysis and Verification Script
========================================

Post-processing tools for ODEM output NetCDF files.

Usage:
    # Full benchmark summary (global total, zonal mean, regional budget)
    python odem_analysis.py --summary output/odem_annual_mean.nc

    # Seasonal cycle from a directory of monthly files
    python odem_analysis.py --seasonal output/

    # All plots for a single-month test run
    python odem_analysis.py --summary output/odem_200601.nc --monthly

    # Compare two runs side by side
    python odem_analysis.py --compare output_a/odem_annual_mean.nc output_b/odem_annual_mean.nc

Author: Metin Baykara
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import netCDF4 as nc4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ===========================================================================
# Dust source regions [label, lon_min, lon_max, lat_min, lat_max]
# ===========================================================================

REGIONS = {
    'Sahara':        (-15,  40,  15, 35),
    'Bodele':        ( 10,  25,  12, 22),
    'W. Africa':     (-15,  10,  10, 20),
    'Middle East':   ( 35,  60,  15, 35),
    'Central Asia':  ( 55,  80,  35, 50),
    'Taklamakan':    ( 75,  95,  35, 45),
    'Gobi':          ( 95, 120,  38, 50),
    'Thar':          ( 65,  78,  22, 32),
    'Horn of Africa':( 38,  55,   5, 18),
    'Patagonia':     (-72, -60, -50,-35),
    'Australia':     (115, 145, -35,-20),
}

# Literature benchmarks for reference lines in plots
# Kok et al. (2021a, ACP) PM20 emission: 5000 ± 1600 Tg/yr global
# ODEM F_d is PM20-equivalent (via C_tune context from Leung 2023).
# With C_tune=0.05, ODEM overshoots PM20 by ~2.5× (consistent with Leung 2023).
BENCHMARK_GLOBAL_PM20_TG = 5000.0          # Kok 2021a PM20, Tg/yr
BENCHMARK_GLOBAL_TG      = BENCHMARK_GLOBAL_PM20_TG    # no conversion needed
BENCHMARK_RANGE = (3400.0, 6600.0)         # ±1σ range

# Dust colormap: white → yellow → orange → red → dark
DUST_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'dust', ['#FFFFFF', '#FFF5CC', '#FFE066', '#FF9933',
             '#FF3300', '#990000', '#330000'], N=256
)


# ===========================================================================
# Core: load output and compute cell areas
# ===========================================================================

def load_mean_field(nc_file):
    """
    Load time-mean dust emission from an ODEM output file.

    Handles:
      - odem_annual_mean.nc  → F_d_mean variable
      - odem_YYYYMM.nc       → F_d variable (time-mean computed here)

    Returns (lon, lat, F_mean [kg/m²/s], meta dict)
    """
    with nc4.Dataset(nc_file, 'r') as ds:
        lon = np.array(ds.variables['lon'][:])
        lat = np.array(ds.variables['lat'][:])

        if 'F_d_mean' in ds.variables:
            F = np.array(ds.variables['F_d_mean'][:], dtype=np.float64)
            period = getattr(ds, 'period', '')
            nt = getattr(ds, 'n_timesteps', None)
        elif 'F_d' in ds.variables:
            raw = np.array(ds.variables['F_d'][:], dtype=np.float64)
            F = np.nanmean(raw, axis=0) if raw.ndim == 3 else raw
            tv = ds.variables.get('time')
            nt = raw.shape[0] if raw.ndim == 3 else 1
            if tv is not None:
                try:
                    times = nc4.num2date(tv[:], tv.units,
                                         getattr(tv, 'calendar', 'standard'))
                    period = f'{times[0]} to {times[-1]}'
                except Exception:
                    period = ''
            else:
                period = ''
        else:
            raise ValueError(f"No F_d or F_d_mean variable in {nc_file}")

        title = getattr(ds, 'title', Path(nc_file).name)

    F = np.nan_to_num(F, nan=0.0)
    F = np.clip(F, 0.0, None)

    meta = {'period': period, 'nt': nt, 'title': title, 'file': nc_file}
    return lon, lat, F, meta


def cell_areas(lat, lon):
    """
    Compute grid cell areas [m²] for a regular lat/lon grid.

    Returns array of shape (nlat, nlon).
    """
    R = 6.371e6  # Earth radius [m]
    dlat = abs(np.median(np.diff(lat)))
    dlon = abs(np.median(np.diff(lon)))
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    areas = R**2 * np.cos(np.deg2rad(lat)) * dlat_rad * dlon_rad
    areas = np.abs(areas)
    return areas[:, np.newaxis] * np.ones((1, len(lon)))


def global_total_tg(F_mean, lat, lon, annualize=True, n_days=30.44):
    """
    Compute global dust emission total [Tg/yr or Tg for period].

    F_mean: time-mean flux [kg/m²/s]
    annualize: if True, scale to annual (365.25 days).
               if False, scale to n_days (default 30.44 = average month).
    n_days: number of days in the period when annualize=False.
    """
    areas = cell_areas(lat, lon)
    total_kg_s = np.sum(F_mean * areas)
    scale = 365.25 * 86400 if annualize else n_days * 86400
    return total_kg_s * scale / 1e9  # Tg


def regional_totals(F_mean, lat, lon):
    """
    Compute annualized dust emission [Tg/yr] for each defined source region.

    Returns dict {region_name: Tg_yr}
    """
    areas = cell_areas(lat, lon)
    lon2d, lat2d = np.meshgrid(lon, lat)
    results = {}

    for name, (lon_min, lon_max, lat_min, lat_max) in REGIONS.items():
        mask = (
            (lon2d >= lon_min) & (lon2d <= lon_max) &
            (lat2d >= lat_min) & (lat2d <= lat_max)
        )
        total = np.sum(F_mean[mask] * areas[mask]) * 365.25 * 86400 / 1e9
        results[name] = total

    return results


# ===========================================================================
# Plot functions
# ===========================================================================

def plot_emission_map(lon, lat, F_mean, meta, outpath=None):
    """Global dust emission map on log scale with source region annotations."""
    fig, ax = plt.subplots(figsize=(16, 8))

    plot_F = F_mean.copy()
    plot_F[plot_F <= 0] = np.nan

    valid = plot_F[np.isfinite(plot_F)]
    if len(valid) == 0:
        print("  WARNING: No active emission cells to plot.")
        return

    vmin = np.nanpercentile(valid, 2)
    vmax = np.nanpercentile(valid, 99.5)
    vmin = max(vmin, 1e-12)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(lon, lat, plot_F, cmap=DUST_CMAP, norm=norm, shading='auto')

    # Annotate source regions
    region_labels = {
        'Bodele':    (17.0, 16.5),
        'W. Africa': (-5.0, 14.0),
        'Middle East': (47.0, 23.0),
        'Taklamakan': (82.0, 39.0),
        'Thar':      (71.0, 27.0),
        'Gobi':      (108.0, 43.0),
        'Patagonia': (-66.0, -43.0),
        'Australia': (130.0, -28.0),
    }
    for name, (rlon, rlat) in region_labels.items():
        ax.plot(rlon, rlat, 'k^', markersize=5, zorder=5)
        ax.annotate(name, (rlon, rlat), fontsize=7, xytext=(4, 4),
                    textcoords='offset points', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('F_d [kg m⁻² s⁻¹]', fontsize=10)

    tg = global_total_tg(F_mean, lat, lon)
    title = f"ODEM — Dust Emission Flux (time mean)\n{meta['period']}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.grid(True, alpha=0.2, linewidth=0.4)

    ax.text(0.01, 0.02, f'Global total: {tg:.0f} Tg/yr  |  Benchmark: ~{BENCHMARK_GLOBAL_TG:.0f} Tg/yr',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {outpath}")
        plt.close(fig)
    else:
        plt.show()


def plot_zonal_mean(lon, lat, F_mean, meta, outpath=None):
    """
    Zonal mean emission profile: latitude vs total emission [Tg/yr/deg].

    Useful for comparing against Kok et al. (2021) Fig. 2 latitudinal distribution.
    """
    areas = cell_areas(lat, lon)
    zonal_kg_s = np.sum(F_mean * areas, axis=1)          # kg/s per latitude band
    zonal_tg = zonal_kg_s * 365.25 * 86400 / 1e9         # Tg/yr per latitude band

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.barh(lat, zonal_tg, height=abs(np.median(np.diff(lat))) * 0.9,
            color='#FF6600', alpha=0.75, edgecolor='none')
    ax.set_xlabel('Emission [Tg yr⁻¹ per latitude band]', fontsize=11)
    ax.set_ylabel('Latitude [°N]', fontsize=11)
    ax.set_title('ODEM — Zonal Mean Dust Emission\n'
                 f'{meta["period"]}', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    # Mark major dust belt
    ax.axhspan(15, 35, alpha=0.08, color='orange', label='Main dust belt (15–35°N)')
    ax.legend(fontsize=9)

    # Summary text
    total = np.sum(zonal_tg)
    nh = np.sum(zonal_tg[lat >= 0])
    sh = np.sum(zonal_tg[lat < 0])
    ax.text(0.97, 0.97,
            f'Total: {total:.0f} Tg/yr\nNH: {nh:.0f}  SH: {sh:.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {outpath}")
        plt.close(fig)
    else:
        plt.show()


def plot_regional_budget(lon, lat, F_mean, meta, outpath=None):
    """
    Bar chart of regional emission totals [Tg/yr] with literature context.
    """
    reg = regional_totals(F_mean, lat, lon)

    # Sort by magnitude
    names = sorted(reg, key=reg.get, reverse=True)
    values = [reg[n] for n in names]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Emission [Tg yr⁻¹]', fontsize=11)
    ax.set_title(f'ODEM — Regional Dust Emission Budget\n{meta["period"]}',
                 fontsize=11, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    ax.text(0.98, 0.97, f'Regions total: {total:.0f} Tg/yr',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {outpath}")
        plt.close(fig)
    else:
        plt.show()


def plot_seasonal_cycle(output_dir, outpath=None):
    """
    Monthly global emission totals [Tg/month] from odem_YYYYMM.nc files.

    Reads all odem_YYYYMM.nc files found in output_dir.
    """
    files = sorted(glob(str(Path(output_dir) / 'odem_??????.nc')))
    if not files:
        print(f"  No odem_YYYYMM.nc files found in {output_dir}")
        return

    months, totals = [], []
    for f in files:
        stem = Path(f).stem  # odem_200601
        ym = stem.split('_')[-1]
        year, month = int(ym[:4]), int(ym[4:6])

        lon, lat, F_mean, meta = load_mean_field(f)

        # Monthly total: F_mean (kg/m²/s) × area × seconds_in_month
        import calendar
        days_in_month = calendar.monthrange(year, month)[1]
        seconds = days_in_month * 86400

        areas = cell_areas(lat, lon)
        total_tg = np.sum(F_mean * areas) * seconds / 1e9
        months.append(f'{year}-{month:02d}')
        totals.append(total_tg)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(months))
    bars = ax.bar(x, totals, color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(months))),
                  edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Emission [Tg month⁻¹]', fontsize=11)
    ax.set_title('ODEM — Seasonal Cycle of Global Dust Emission', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    annual = sum(totals)
    ax.text(0.98, 0.97, f'Annual total: {annual:.0f} Tg/yr\nBenchmark: ~{BENCHMARK_GLOBAL_TG:.0f} Tg/yr',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {outpath}")
        plt.close(fig)
    else:
        plt.show()

    return dict(zip(months, totals))


def benchmark_summary(nc_file, monthly=False):
    """
    Print a full benchmark summary for a model output file.

    monthly: if True, scale global total to monthly instead of annual.
    """
    lon, lat, F_mean, meta = load_mean_field(nc_file)
    tg = global_total_tg(F_mean, lat, lon, annualize=not monthly)
    reg = regional_totals(F_mean, lat, lon)

    active = F_mean[F_mean > 0]
    n_active = len(active)
    n_total = F_mean.size

    print()
    print("=" * 60)
    print("  ODEM — Benchmark Summary")
    print("=" * 60)
    print(f"  File:    {Path(nc_file).name}")
    print(f"  Period:  {meta['period']}")
    print()
    print("  GLOBAL TOTAL")
    label = "month" if monthly else "yr"
    print(f"    ODEM:       {tg:>8.0f} Tg/{label}")
    if not monthly:
        print(f"    Kok 2021 PM20:     {BENCHMARK_GLOBAL_PM20_TG:>8.0f} Tg/yr  (Kok 2021a global)")
        print(f"    PM20 range:        {BENCHMARK_RANGE[0]:>8.0f}–{BENCHMARK_RANGE[1]:.0f} Tg/yr")
        ratio = tg / BENCHMARK_GLOBAL_TG
        print(f"    Ratio ODEM/target: {ratio:>8.2f}")
    print()
    print("  EMISSION FIELD")
    print(f"    Active cells:  {n_active:>7,} / {n_total:,} ({100*n_active/n_total:.1f}%)")
    if n_active > 0:
        print(f"    Mean (active): {np.mean(active):.3e} kg/m²/s")
        print(f"    Median:        {np.median(active):.3e} kg/m²/s")
        print(f"    P95:           {np.percentile(active, 95):.3e} kg/m²/s")
        print(f"    Max:           {np.max(active):.3e} kg/m²/s")
    print()
    print("  REGIONAL BUDGET  [Tg/yr, annualized]")
    total_regional = sum(reg.values())
    for name in sorted(reg, key=reg.get, reverse=True):
        pct = 100 * reg[name] / tg if tg > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"    {name:<18} {reg[name]:6.1f} Tg/yr  ({pct:4.1f}%)  {bar}")
    print(f"    {'(regions sum)':<18} {total_regional:6.1f} Tg/yr")
    print()

    # Hemispheric split
    areas = cell_areas(lat, lon)
    lat2d = np.broadcast_to(lat[:, np.newaxis], F_mean.shape)
    nh_tg = np.sum(F_mean[lat2d >= 0] * areas[lat2d >= 0]) * 365.25 * 86400 / 1e9
    sh_tg = np.sum(F_mean[lat2d < 0] * areas[lat2d < 0]) * 365.25 * 86400 / 1e9
    print(f"  HEMISPHERIC SPLIT  [annualized]")
    print(f"    NH: {nh_tg:.0f} Tg/yr ({100*nh_tg/(nh_tg+sh_tg+1e-9):.1f}%)")
    print(f"    SH: {sh_tg:.0f} Tg/yr ({100*sh_tg/(nh_tg+sh_tg+1e-9):.1f}%)")
    print("=" * 60)
    print()

    return {'global_tg': tg, 'regions': reg}


def run_all(nc_file, outdir, monthly=False):
    """Generate all analysis plots + benchmark summary for one output file."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lon, lat, F_mean, meta = load_mean_field(nc_file)
    stem = Path(nc_file).stem

    print(f"\nRunning ODEM analysis: {Path(nc_file).name}")
    print(f"Output directory: {outdir}")

    benchmark_summary(nc_file, monthly=monthly)

    plot_emission_map(lon, lat, F_mean, meta,
                      outpath=str(outdir / f'{stem}_map.png'))
    plot_zonal_mean(lon, lat, F_mean, meta,
                    outpath=str(outdir / f'{stem}_zonal.png'))
    plot_regional_budget(lon, lat, F_mean, meta,
                         outpath=str(outdir / f'{stem}_regions.png'))

    print(f"\nDone. 3 plots saved to {outdir}/")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='ODEM — Analysis and Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full summary for annual mean output
  python odem_analysis.py --summary output/odem_annual_mean.nc

  # Single-month test run (scales as monthly, not annual)
  python odem_analysis.py --summary output/odem_200601.nc --monthly

  # Seasonal cycle from full-year run
  python odem_analysis.py --seasonal output/

  # Compare two runs
  python odem_analysis.py --compare output_a/odem_annual_mean.nc output_b/odem_annual_mean.nc
""")

    p.add_argument('--summary', metavar='NC_FILE',
                   help='Run full analysis (map + zonal + regional + benchmark) on one file')
    p.add_argument('--monthly', action='store_true',
                   help='Treat input as single-month (do not annualize global total)')
    p.add_argument('--seasonal', metavar='OUTPUT_DIR',
                   help='Plot seasonal cycle from directory of odem_YYYYMM.nc files')
    p.add_argument('--compare', nargs=2, metavar=('FILE_A', 'FILE_B'),
                   help='Compare two annual mean files')
    p.add_argument('--outdir', default='./odem_plots',
                   help='Output directory for plots (default: ./odem_plots)')

    args = p.parse_args()

    if args.summary:
        run_all(args.summary, args.outdir, monthly=args.monthly)

    if args.seasonal:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        data = plot_seasonal_cycle(args.seasonal,
                                   outpath=str(out / 'seasonal_cycle.png'))
        if data:
            print(f"Monthly totals: {data}")

    if args.compare:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        lon_a, lat_a, F_a, meta_a = load_mean_field(args.compare[0])
        lon_b, lat_b, F_b, meta_b = load_mean_field(args.compare[1])

        tg_a = global_total_tg(F_a, lat_a, lon_a)
        tg_b = global_total_tg(F_b, lat_b, lon_b)
        print(f"\n  Run A: {tg_a:.0f} Tg/yr  ({Path(args.compare[0]).name})")
        print(f"  Run B: {tg_b:.0f} Tg/yr  ({Path(args.compare[1]).name})")
        print(f"  Ratio A/B: {tg_a/tg_b:.3f}")

        if np.array_equal(lon_a, lon_b) and np.array_equal(lat_a, lat_b):
            both = (F_a > 0) & (F_b > 0)
            if both.sum() > 0:
                r = np.corrcoef(F_a[both], F_b[both])[0, 1]
                print(f"  Spatial correlation (active cells): r = {r:.4f}")

        # Side-by-side map
        valid_all = np.concatenate([F_a[F_a > 0], F_b[F_b > 0]])
        if len(valid_all) > 0:
            vmin = max(np.percentile(valid_all, 2), 1e-12)
            vmax = np.percentile(valid_all, 99.5)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            for ax, F, lon_grid, lat_grid, meta, tg in [
                (axes[0], F_a, lon_a, lat_a, meta_a, tg_a),
                (axes[1], F_b, lon_b, lat_b, meta_b, tg_b),
            ]:
                plot_F = F.copy()
                plot_F[plot_F <= 0] = np.nan
                im = ax.pcolormesh(lon_grid, lat_grid, plot_F, cmap=DUST_CMAP,
                                   norm=norm, shading='auto')
                ax.set_title(f"{Path(meta['file']).name}\n{tg:.0f} Tg/yr",
                             fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.2)
                plt.colorbar(im, ax=ax, shrink=0.7, label='kg m⁻² s⁻¹')

            plt.suptitle('ODEM — Run Comparison', fontsize=13, fontweight='bold')
            plt.tight_layout()
            outpath = str(out / 'comparison_map.png')
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {outpath}")
            plt.close(fig)

    if not any([args.summary, args.seasonal, args.compare]):
        p.print_help()


if __name__ == '__main__':
    main()
