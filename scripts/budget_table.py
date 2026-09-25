#!/usr/bin/env python
"""Global and regional annual budgets for a set of ODEM run directories.

Reads each run's odem_annual_mean.nc (F_d_mean, kg m-2 s-1) and, when present,
the monthly odem_YYYYMM.nc files for the seasonal cycle. Prints a Markdown
table and writes a CSV: global total (Tg/yr), ratio to the Kok et al. (2021)
PM20 constraint, the seven paper regions of Sect. 4.3 with their ratio to the
matching Kok et al. (2021) region, and the monthly totals.

Annual-total convention (the one of v1, odem.py and odem_regression.py):
the time mean of the flux over every output step, times the spherical cell
area R^2 cos(lat) dlat dlon (R = 6.371e6 m), times 365.25 days. Monthly
totals use the same cell area and the seconds covered by the month's steps.
Every script that reports a total imports annual_total_tg / cell_area from
here.

Usage:
  python scripts/budget_table.py --csv table.csv <run_dir> [<run_dir> ...]
A run directory may also be given as label=path.
"""
import argparse
import csv
import glob
import os

import numpy as np
import netCDF4 as nc

# Region boxes as in the submitted manuscript (papers/odem_gmd/scripts/plot_f06_regional_budget.py)
REGIONS = {
    "N. Africa":   {"lat": (0, 40),    "lon": (-20, 40)},
    "Middle East": {"lat": (12, 45),   "lon": (40, 65)},
    "East Asia":   {"lat": (25, 50),   "lon": (70, 130)},
    "N. America":  {"lat": (15, 50),   "lon": (-120, -80)},
    "S. America":  {"lat": (-56, 0),   "lon": (-80, -35)},
    "S. Africa":   {"lat": (-35, 0),   "lon": (10, 50)},
    "Australia":   {"lat": (-45, -10), "lon": (110, 155)},
}
# Kok et al. (2021), ACP 21, 8169, Table 1, inverse model, annual PM20 dust
# emission, median and 1-sigma range [Tg/yr] (printed as 10^3 Tg/yr). Keyed by the
# ODEM box; the Kok region each box is compared with is named in KOK2021_NAME.
# The boxes are not Kok's regions; the mapping is stated in the paper (Sect. 3).
KOK2021 = {
    "N. Africa":   (2100, 1600, 4300),
    "Middle East": (1380, 970, 2590),
    "East Asia":   (580, 420, 1120),
    "N. America":  (130, 50, 240),
    "S. America":  (190, 100, 350),
    "S. Africa":   (100, 60, 190),
    "Australia":   (160, 90, 290),
    "Global":      (4700, 3400, 9100),
}
KOK2021_NAME = {
    "N. Africa":   "all of North Africa",
    "Middle East": "Middle East/central Asia",
    "East Asia":   "East Asia",
    "N. America":  "North America",
    "S. America":  "South America",
    "S. Africa":   "southern Africa",
    "Australia":   "Australia",
    "Global":      "all source regions",
}
SEC_PER_YR = 365.25 * 24 * 3600


def cell_area(lat, lon):
    dlat = abs(lat[1] - lat[0]); dlon = abs(lon[1] - lon[0])
    cell = (dlat * np.pi / 180) * (dlon * np.pi / 180) * 6.371e6 ** 2 * np.cos(np.deg2rad(lat))
    return cell[:, None] * np.ones((1, len(lon)))


def annual_total_tg(F_mean, lat, lon, mask=None):
    """Annual total [Tg/yr] of a time-mean flux F_mean [kg m-2 s-1] (the one convention)."""
    F = np.nan_to_num(np.asarray(F_mean, dtype=np.float64))
    A = cell_area(lat, lon)
    if mask is not None:
        F = F[mask]; A = A[mask]
    return float((F * A).sum() * SEC_PER_YR / 1e9)


def region_mask(lat, lon, box):
    la = (lat >= box['lat'][0]) & (lat <= box['lat'][1])
    lo = (lon >= box['lon'][0]) & (lon <= box['lon'][1])
    return la[:, None] & lo[None, :]


def budgets(path):
    with nc.Dataset(path) as d:
        lat = np.array(d.variables['lat'][:]); lon = np.array(d.variables['lon'][:])
        F = np.nan_to_num(np.array(d.variables['F_d_mean'][:], dtype=np.float64))
        meta = {k: str(d.getncattr(k)) for k in d.ncattrs()}
    out = {'Global': annual_total_tg(F, lat, lon)}
    for name, box in REGIONS.items():
        out[name] = annual_total_tg(F, lat, lon, region_mask(lat, lon, box))
    return out, meta, (lat, lon)


def monthly_totals(run_dir):
    """Tg per month from odem_YYYYMM.nc files (mean flux x area x seconds in the file)."""
    res = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'odem_[0-9][0-9][0-9][0-9][0-9][0-9].nc'))):
        with nc.Dataset(f) as d:
            lat = np.array(d.variables['lat'][:]); lon = np.array(d.variables['lon'][:])
            tv = d.variables['time']
            t = nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
            F = np.nan_to_num(np.array(d.variables['F_d'][:], dtype=np.float64)).mean(axis=0)
        n = len(t)
        # month length in seconds from the stamps (uniform step assumed)
        step = (t[1] - t[0]).total_seconds() if n > 1 else 3600.0
        month = f'{t[0].year}-{t[0].month:02d}'
        res[month] = float((F * cell_area(lat, lon)).sum() * step * n / 1e9)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='run directories, optionally label=path')
    p.add_argument('--csv', default=None)
    p.add_argument('--no-monthly', action='store_true')
    a = p.parse_args()
    rows = []
    for spec in a.runs:
        label, path = spec.split('=', 1) if '=' in spec else (os.path.basename(spec.rstrip('/\\')), spec)
        am = os.path.join(path, 'odem_annual_mean.nc')
        if not os.path.exists(am):
            print(f'skip {label}: no odem_annual_mean.nc'); continue
        b, meta, _ = budgets(am)
        row = {'run': label, 'scheme': meta.get('scheme', ''), 'source': meta.get('source', ''),
               'n_timesteps': meta.get('n_timesteps', ''), 'period': meta.get('period', '')}
        for k, v in b.items():
            row[k] = v
            row[k + ' /Kok'] = v / KOK2021[k][0]
        if not a.no_monthly:
            for m, v in monthly_totals(path).items():
                row[m] = v
        rows.append(row)
    if not rows:
        return
    regions = ['Global'] + list(REGIONS)
    print(f"Constraint: Kok et al. (2021) Table 1, inverse model, PM20 emission, median (1-sigma range), Tg/yr: "
          + '; '.join(f"{r} vs {KOK2021_NAME[r]} {KOK2021[r][0]} ({KOK2021[r][1]} to {KOK2021[r][2]})" for r in regions) + '\n')
    print(f"| run | scheme | Global Tg/yr | x Kok({KOK2021['Global'][0]}) | " + ' | '.join(f'{r} ({KOK2021[r][0]})' for r in regions[1:]) + ' |')
    print('|' + '---|' * (4 + len(regions) - 1))
    for r in rows:
        cells = [r['run'], r['scheme'], f"{r['Global']:.0f}", f"{r['Global /Kok']:.2f}"]
        cells += [f"{r[k]:.0f} ({r[k + ' /Kok']:.1f}x)" for k in regions[1:]]
        print('| ' + ' | '.join(cells) + ' |')
    months = sorted({k for r in rows for k in r if k[:1].isdigit()})
    if months:
        print('\nMonthly totals (Tg):')
        print('| run | ' + ' | '.join(months) + ' |')
        print('|' + '---|' * (1 + len(months)))
        for r in rows:
            print('| ' + r['run'] + ' | ' + ' | '.join(f"{r.get(m, float('nan')):.0f}" for m in months) + ' |')
    if a.csv:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
