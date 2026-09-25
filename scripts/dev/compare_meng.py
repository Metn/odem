"""Compare ODEM DEAD runs on the MERRA-2 grid against the Meng et al. (2021)
offline DEAD emissions (GEOS-Chem HEMCO OFFLINE_DUST/v2021-08, MERRA-2 native
grid, sum of the four transport bins, already carrying T = 4.7376e-4).

Totals follow the one annual-total convention of scripts/budget_table.py (v1's):
time-mean flux x spherical cell area x 365.25 days, applied to ODEM and Meng
alike. The time mean over the months covered weights each month by its length
in seconds. R = 0.9188 x ODEM / Meng (ODEM is the full vertical flux, Meng the
four transport bins).

Usage:
    python scripts/dev/compare_meng.py --months 1-12 RUN_DIR [RUN_DIR ...]
    python scripts/dev/compare_meng.py --months 1-12 --by-month --by-zone RUN_DIR

RUN_DIR holds odem_2006MM.nc (and optionally odem_diag_mean.nc), or
months/mMM/odem_2006MM.nc as written by scripts/run_year.py. The Meng monthly
mean is cached as meng_2006MM_mean.npz under --cache.
"""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import argparse, calendar, glob, os
from pathlib import Path
import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget_table import annual_total_tg, cell_area, region_mask  # noqa: E402

R = os.environ.get('ODEM_DATA_ROOT')
if not R:
    raise SystemExit('set ODEM_DATA_ROOT to the ODEM data directory (soilgrids/, prigent_data/, modis_lai/, ...)')
MENG_DIR = f"{R}/dust_inputs/offline_dust_geoschem/v2021-08_DEAD"
BIN_FRAC = 0.9187803392840019   # 4 GEOS-Chem transport bins / total DEAD vertical flux
YEAR = 2006
# Zones of the by-zone table (as in the 2026-09-24 comparison)
ZONES = {
    "Sahara 20-35N,20W-40E":    {"lat": (20, 35), "lon": (-20, 40)},
    "Sahel 10-20N,20W-40E":     {"lat": (10, 20), "lon": (-20, 40)},
    "Arabia/ME 12-35N,40-65E":  {"lat": (12, 35), "lon": (40, 65)},
    "C/E Asia 35-50N,60-120E":  {"lat": (35, 50), "lon": (60, 120)},
    "Australia 10-45S,110-155E": {"lat": (-45, -10), "lon": (110, 155)},
}


def parse_months(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def meng_month(month, cache):
    """Time-mean of the Meng offline DEAD flux (sum of 4 bins), kg m-2 s-1."""
    f = Path(cache) / f"meng_{YEAR}{month:02d}_mean.npz"
    if f.exists():
        m = np.load(f)
        return m["mean"], m["lat"], m["lon"]
    files = sorted(glob.glob(f"{MENG_DIR}/{month:02d}/dust_emissions_05.{YEAR}{month:02d}*.nc"))
    if not files:
        raise FileNotFoundError(f"no Meng files for {YEAR}-{month:02d} in {MENG_DIR}")
    acc = None; n = 0
    for fn in files:
        with nc.Dataset(fn) as ds:
            if acc is None:
                area = np.array(ds.variables['AREA'][:], dtype=np.float64)
                lat = np.array(ds.variables['lat'][:]); lon = np.array(ds.variables['lon'][:])
                acc = np.zeros(area.shape)
            for b in range(1, 5):
                v = np.array(ds.variables[f'EMIS_DST{b}'][:], dtype=np.float64)
                v = np.where(np.isfinite(v) & (np.abs(v) < 1e10), v, 0.0)
                acc += v.sum(axis=0)
            n += ds.variables['time'].shape[0]
    mean = acc / n
    np.savez(f, mean=mean, area=area, lat=lat, lon=lon, n=n)
    return mean, lat, lon


def run_month(run_dir, month):
    """Time-mean ODEM F_d for one month, kg m-2 s-1, plus the scheme attr."""
    d = Path(run_dir)
    for cand in (d / f"odem_{YEAR}{month:02d}.nc", d / "months" / f"m{month:02d}" / f"odem_{YEAR}{month:02d}.nc"):
        if cand.exists():
            break
    else:
        raise FileNotFoundError(f"no odem_{YEAR}{month:02d}.nc under {d}")
    with nc.Dataset(cand) as ds:
        nt = ds.variables["F_d"].shape[0]
        lat = np.array(ds.variables["lat"][:]); lon = np.array(ds.variables["lon"][:])
        F = np.zeros((lat.size, lon.size))
        for t0 in range(0, nt, 48):
            F += np.nan_to_num(np.array(ds.variables["F_d"][t0:t0 + 48], dtype=np.float64)).sum(axis=0)
        return F / nt, lat, lon, getattr(ds, "scheme", "?")


def month_tg(F, lat, lon, seconds, mask=None):
    """Tg emitted in a month: monthly-mean flux x the same cell area x the month's seconds."""
    A = cell_area(lat, lon)
    if mask is not None:
        F = F[mask]; A = A[mask]
    return float((F * A).sum() * seconds / 1e9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", default="1")
    ap.add_argument("--cache", default=f"{R}/output_v2_dev")
    ap.add_argument("--by-month", action="store_true", help="R and r over land for every month")
    ap.add_argument("--by-zone", action="store_true", help="R by zone and month")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    months = parse_months(a.months)
    seconds = {m: calendar.monthrange(YEAR, m)[1] * 86400.0 for m in months}
    S = sum(seconds.values())

    meng = {}
    for m in months:
        meng[m] = meng_month(m, a.cache)
    lat, lon = meng[months[0]][1], meng[months[0]][2]
    meng_ann = sum(meng[m][0] * seconds[m] for m in months) / S     # time-mean flux over the months
    meng_tot = annual_total_tg(meng_ann, lat, lon)
    print(f"Convention: time-mean flux x spherical cell area x 365.25 d (scripts/budget_table.py).")
    print(f"Months {months}. Meng v2021-08 (carries T): {meng_tot:.1f} Tg/yr")

    for run in a.runs:
        Fm = {}
        scheme = None
        for m in months:
            F, olat, olon, scheme = run_month(run, m)
            assert np.allclose(olat, lat) and np.allclose(olon, lon), f"grid mismatch in {run}"
            Fm[m] = F
        ann = sum(Fm[m] * seconds[m] for m in months) / S
        tot = annual_total_tg(ann, lat, lon)
        acc = ann * S
        meng_sum = meng_ann * S
        diag = next((p for p in (Path(run) / "odem_diag_mean.nc",
                                 Path(run) / "months" / f"m{months[0]:02d}" / "odem_diag_mean.nc") if p.exists()), None)
        if diag is not None:
            with nc.Dataset(diag) as ds:
                land = np.array(ds.variables["lsm"][:]) > 0.5
            where = "land"
        else:
            land = np.ones_like(acc, dtype=bool); where = "all cells"
        r = float(np.corrcoef(acc[land].ravel(), meng_sum[land].ravel())[0, 1])
        both = (acc > 0) & (meng_sum > 0)
        rlog = float(np.corrcoef(np.log10(acc[both]), np.log10(meng_sum[both]))[0, 1]) if both.sum() > 2 else float('nan')
        print(f"\n{Path(run).name}  [{scheme}]")
        print(f"  total {tot:.1f} Tg/yr; ratio to Meng {tot / meng_tot:.3f}; x bin fraction (R) {tot * BIN_FRAC / meng_tot:.3f}")
        print(f"  Pearson r over {where} {r:.3f}; log10 r where both > 0 {rlog:.3f} (n={int(both.sum())}); "
              f"active cells ODEM {int((acc > 0).sum())}, Meng {int((meng_sum > 0).sum())}, overlap {int(both.sum())}")

        if a.by_month:
            print(f"\n  R = 0.9188 x ODEM / Meng by month, r over {where}")
            print("  month  ODEM Tg  Meng Tg      R      r")
            for m in months:
                o = month_tg(Fm[m], lat, lon, seconds[m]); g = month_tg(meng[m][0], lat, lon, seconds[m])
                rm = float(np.corrcoef(Fm[m][land].ravel(), meng[m][0][land].ravel())[0, 1])
                print(f"  {m:5d} {o:8.1f} {g:8.1f} {BIN_FRAC * o / g:6.3f} {rm:6.3f}")

        if a.by_zone:
            masks = {z: region_mask(lat, lon, box) for z, box in ZONES.items()}
            print("\n  R = 0.9188 x ODEM / Meng by zone and month (Meng Tg in the month in parentheses)")
            print("  month " + " | ".join(f"{z:>26s}" for z in ZONES))
            for m in months:
                cells = []
                for z, mk in masks.items():
                    o = month_tg(Fm[m], lat, lon, seconds[m], mk); g = month_tg(meng[m][0], lat, lon, seconds[m], mk)
                    cells.append(f"{BIN_FRAC * o / g:9.3f} ({g:7.1f} Tg)".rjust(26) if g > 0 else f"{'n/a':>26s}")
                print(f"  {m:5d} " + " | ".join(cells))


if __name__ == "__main__":
    main()
