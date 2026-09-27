#!/usr/bin/env python
"""ODEM regression harness.

Step zero before any refactor. The physics suite in odem_verify.py checks ODEM
against external constraints (analytical thresholds, the Kok 2021 PM20 budget).
It does NOT check ODEM against ITSELF, so a refactor could shift the published
global totals by a few percent with every existing test still passing.

This harness closes that gap with two assertions:

  1. BUDGET   the global annual total recomputed from each archived annual-mean
              file still equals the pinned value
  2. BITWISE  a freshly computed month is bit-for-bit identical to the archived
              month it reproduces

Reproduction was demonstrated on 2026-08-13: a fresh run of released v1.0 on the
archived MERRA-2 January 2006 forcing reproduced the published monthly flux field
exactly, array sum 6.555398e-02 over all 744 x 361 x 576 values.

Usage
-----
  python odem_regression.py --pin   --archive-root <data root>
  python odem_regression.py --check --archive-root <data root>
  python odem_regression.py --check --skip-missing --archive-root <unpacked data deposit>
  python odem_regression.py --compare-month new/odem_200601.nc ref/odem_200601.nc

--pin writes odem_regression_pins.json. Regenerate it only deliberately, and never
as a way of making a failing check pass.
"""
import argparse
import hashlib
import json
import os
import sys
import io

import numpy as np
import netCDF4 as nc

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "odem_regression_pins.json")

# Archived runs that define the published results. Directory names are relative
# to --archive-root. Keep this list in sync with the manuscript's Table of runs.
RUNS = [
    "output_2006",                      # ERA5, 3-hourly, full year
    "output_2006_era5_hourly",          # ERA5, 1-hourly, full year
    "output_2006_merra2",               # MERRA-2, 1-hourly, full year
    "output_2006_ctune020",             # as above but C_tune = 0.020
    "output_2006_era5_hourly_ctune020",
    "output_2006_merra2_ctune020",
    # Development-tree run, NOT in the manuscript: Chappell albedo drag partition
    # (compute_emission_aem), ERA5 3-hourly grid, January 2006 only (248 stamps),
    # so its "annual" total is January annualised. It is the pin for reinstating
    # the AEM scheme in v2: --compare-month against its odem_200601.nc must be
    # bitwise. Added 2026-09-24.
    "output_2006_aem",
    # v1.1 primaries on the MERRA-2 grid (2006, primary statics: conservative,
    # z0a in log space; DEAD on the matched 2x2.5 maps). The published v1.1
    # totals; pinned at the matrix freeze, 2026-09-25.
    "output_v2/k14_m2_consl_2006",
    "output_v2/aem_m2_cons_2006",
    "output_v2/dead_g_m2_consl_2006",
    "output_v2/dead_z_m2_consl_2006",
    # Sensitivity row: AEM on MERRA-2 with the LeGrand et al. (2023) sandblasting
    # term (--aem-sandblasting legrand), pinned 2026-09-25.
    "output_v2/aem_lg_m2_cons_2006",
]

# The three ctune020 runs are exact 0.4x scalings of their C_tune = 0.05
# counterparts (0.020 / 0.05), reproduced to every digit on 2026-08-13. That is
# an ODEM-internal demonstration that C_tune enters as a linear multiplier, which
# is the substance of the reply to RC1 point 3.

# Relative tolerance on the recomputed budget. Tight on purpose: this is a
# same-inputs, same-code comparison, so anything above float noise is a change
# in behaviour and should be looked at, not absorbed.
RTOL = 1e-6

# File name of each run's annual mean in the flat Zenodo data deposit
# (v1.0 data deposit: 10.5281/zenodo.19320862; byte-identical to the archived files). When
# <archive-root>/<run>/odem_annual_mean.nc is absent the harness looks for
# <archive-root>/<name> instead, so --archive-root may point at the unpacked
# deposit. The v1.1 primaries follow odem_annual_mean_<run directory>.nc.
DEPOSIT_NAMES = {
    "output_2006":                      "odem_annual_mean_era5_3h_ctune005.nc",
    "output_2006_era5_hourly":          "odem_annual_mean_era5_1h_ctune005.nc",
    "output_2006_merra2":               "odem_annual_mean_merra2_ctune005.nc",
    "output_2006_ctune020":             "odem_annual_mean_era5_3h_ctune020.nc",
    "output_2006_era5_hourly_ctune020": "odem_annual_mean_era5_1h_ctune020.nc",
    "output_2006_merra2_ctune020":      "odem_annual_mean_merra2_ctune020.nc",
}


def annual_mean_path(root, run):
    """The run's annual-mean file under root: run directory first, then the deposit name."""
    p = os.path.join(root, run, "odem_annual_mean.nc")
    if os.path.exists(p):
        return p
    name = DEPOSIT_NAMES.get(run, f"odem_annual_mean_{os.path.basename(run)}.nc")
    return os.path.join(root, name)


def global_total_tg_per_year(path):
    """Recompute the annual global budget from an annual-mean file.

    Uses the same spherical cell-area formula as run_model in odem.py, so the
    number is comparable with what the model prints at the end of a run.
    """
    with nc.Dataset(path) as d:
        lat = np.array(d.variables["lat"][:])
        lon = np.array(d.variables["lon"][:])
        flux = np.nan_to_num(np.array(d.variables["F_d_mean"][:]))
        meta = {k: str(d.getncattr(k)) for k in d.ncattrs()}

    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    cell = (dlat * np.pi / 180) * (dlon * np.pi / 180) * 6.371e6 ** 2 * np.cos(np.deg2rad(lat))
    area = cell[:, np.newaxis] * np.ones((1, len(lon)))

    total_tg = float((flux * area).sum() * 365.25 * 24 * 3600 / 1e9)
    return total_tg, {
        "shape": list(flux.shape),
        "n_timesteps": meta.get("n_timesteps", "?"),
        "period": meta.get("period", "?"),
    }


def model_fingerprint():
    """SHA-256 of odem.py with line endings normalised to LF, so a pin file
    records which model produced it and the same source gives the same
    fingerprint on every platform (git checkouts with core.autocrlf differ
    only in CRLF versus LF)."""
    p = os.path.join(HERE, "odem.py")
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16]


def cmd_pin(root):
    pins = {"model_sha256_16": model_fingerprint(), "runs": {}}
    for run in RUNS:
        path = annual_mean_path(root, run)
        if not os.path.exists(path):
            print(f"  skip  {run}: no annual-mean file")
            continue
        total, meta = global_total_tg_per_year(path)
        pins["runs"][run] = {"total_tg_per_year": total, **meta}
        print(f"  pin   {run:<34} {total:9.1f} Tg/yr  nt={meta['n_timesteps']}")
    with open(PINS, "w", encoding="utf-8") as fh:
        json.dump(pins, fh, indent=2)
    print(f"\nwrote {PINS}")
    return 0


def cmd_check(root, skip_missing=False):
    if not os.path.exists(PINS):
        print(f"no pin file at {PINS}; run --pin first")
        return 2
    pins = json.load(open(PINS, encoding="utf-8"))
    now = model_fingerprint()
    if pins.get("model_sha256_16") != now:
        print(f"note: odem.py has changed since pinning "
              f"({pins.get('model_sha256_16')} -> {now}). That is the point of this check.\n")

    failures = 0; missing = 0
    for run, expect in pins["runs"].items():
        path = annual_mean_path(root, run)
        if not os.path.exists(path):
            print(f"  MISS  {run}: archived file absent")
            missing += 1
            failures += 0 if skip_missing else 1
            continue
        total, meta = global_total_tg_per_year(path)
        want = expect["total_tg_per_year"]
        rel = abs(total - want) / want if want else 0.0
        ok = rel <= RTOL and meta["shape"] == expect["shape"]
        print(f"  {'ok  ' if ok else 'FAIL'}  {run:<34} {total:9.1f} vs {want:9.1f} Tg/yr  rel={rel:.2e}")
        failures += 0 if ok else 1

    checked = len(pins["runs"]) - (missing if skip_missing else 0)
    print(f"\n{checked - failures} of {checked} runs match"
          + (f" ({missing} absent, skipped)" if skip_missing and missing else ""))
    return 1 if failures else 0


def cmd_compare_month(new_path, ref_path):
    """Bitwise comparison of two monthly flux files."""
    with nc.Dataset(new_path) as a, nc.Dataset(ref_path) as b:
        A = np.nan_to_num(np.array(a.variables["F_d"][:]))
        B = np.nan_to_num(np.array(b.variables["F_d"][:]))

    if A.shape != B.shape:
        print(f"FAIL shape {A.shape} vs {B.shape}")
        return 1

    if np.array_equal(A, B):
        print(f"ok   bitwise identical over {A.size} values, sum {A.sum():.6e}")
        return 0

    diff = np.abs(A - B)
    nzero = B > 0
    print(f"FAIL {int((diff > 0).sum())} of {diff.size} values differ")
    print(f"     max abs {diff.max():.3e}   max rel {(diff[nzero] / B[nzero]).max():.3e}"
          if nzero.any() else f"     max abs {diff.max():.3e}")
    print(f"     sum new {A.sum():.6e}  sum ref {B.sum():.6e}  ratio {A.sum() / B.sum():.9f}")
    return 1


def main():
    p = argparse.ArgumentParser(description="ODEM regression harness")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pin", action="store_true", help="write pins from the archived runs")
    g.add_argument("--check", action="store_true", help="verify archived runs against the pins")
    g.add_argument("--compare-month", nargs=2, metavar=("NEW", "REF"),
                   help="bitwise-compare two monthly flux files")
    p.add_argument("--archive-root", default=os.environ.get("ODEM_DATA", ""),
                   help="directory holding the run directories, or the unpacked flat data "
                        "deposit (default: $ODEM_DATA)")
    p.add_argument("--skip-missing", action="store_true",
                   help="with --check: runs absent under --archive-root are reported, not failed "
                        "(for a deposit that holds only some of the pinned runs)")
    args = p.parse_args()

    if args.compare_month:
        return cmd_compare_month(*args.compare_month)
    if not args.archive_root:
        print("--archive-root is required (or set ODEM_DATA)")
        return 2
    return cmd_pin(args.archive_root) if args.pin else cmd_check(args.archive_root, args.skip_missing)


if __name__ == "__main__":
    sys.exit(main())
