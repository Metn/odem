"""Compare ODEM DEAD runs on the MERRA-2 grid against the Meng et al. (2021)
offline DEAD emissions (GEOS-Chem HEMCO OFFLINE_DUST/v2021-08, MERRA-2 native
grid, sum of the four transport bins, already carrying T = 4.7376e-4).

Usage:
    python scripts/dev/compare_meng.py --months 1 RUN_DIR [RUN_DIR ...]
    python scripts/dev/compare_meng.py --months 1-12 RUN_DIR ...

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

R = os.environ.get('ODEM_DATA_ROOT')
if not R:
    raise SystemExit('set ODEM_DATA_ROOT to the ODEM data directory (soilgrids/, prigent_data/, modis_lai/, ...)')
MENG_DIR = f"{R}/dust_inputs/offline_dust_geoschem/v2021-08_DEAD"
BIN_FRAC = 0.9187803392840019   # 4 GEOS-Chem transport bins / total DEAD vertical flux
YEAR = 2006


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
        return m["mean"], m["area"], m["lat"], m["lon"], int(m["n"])
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
    return mean, area, lat, lon, n


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
            F += np.array(ds.variables["F_d"][t0:t0 + 48], dtype=np.float64).sum(axis=0)
        return F / nt, lat, lon, getattr(ds, "scheme", "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", default="1")
    ap.add_argument("--cache", default=f"{R}/output_v2_dev")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    months = parse_months(a.months)

    seconds = {m: calendar.monthrange(YEAR, m)[1] * 86400.0 for m in months}
    meng_tot = 0.0
    meng_sum = None
    for m in months:
        mean, area, lat, lon, _ = meng_month(m, a.cache)
        meng_tot += float((mean * area).sum()) * seconds[m]
        meng_sum = mean * seconds[m] if meng_sum is None else meng_sum + mean * seconds[m]
    scale = 365.25 * 86400.0 / sum(seconds.values())   # annualise the months covered
    print(f"Months {months}. Meng v2021-08 (carries T): {meng_tot / 1e9:.2f} Tg over the period, "
          f"{meng_tot / 1e9 * scale:.1f} Tg/yr annualised")

    for run in a.runs:
        tot = 0.0; acc = None; scheme = None
        for m in months:
            F, olat, olon, scheme = run_month(run, m)
            assert np.allclose(olat, lat) and np.allclose(olon, lon), f"grid mismatch in {run}"
            tot += float((F * area).sum()) * seconds[m]
            acc = F * seconds[m] if acc is None else acc + F * seconds[m]
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
        print(f"\n{run}  [{scheme}]")
        print(f"  total {tot / 1e9:.2f} Tg ({tot / 1e9 * scale:.1f} Tg/yr annualised); "
              f"ratio to Meng {tot / meng_tot:.3f}; x bin fraction {tot * BIN_FRAC / meng_tot:.3f}")
        print(f"  Pearson r over {where} {r:.3f}; log10 r where both > 0 {rlog:.3f} (n={int(both.sum())}); "
              f"active cells ODEM {int((acc > 0).sum())}, Meng {int((meng_sum > 0).sum())}, overlap {int(both.sum())}")


if __name__ == "__main__":
    main()
