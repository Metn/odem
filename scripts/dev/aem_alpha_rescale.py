"""Expected totals of an AEM run with the LeGrand et al. (2023) sandblasting term,
from an existing AEM run with the default term (pre-registration of the
sensitivity run, 2026-09-25).

alpha enters the AEM vertical flux as a per-cell multiplier outside every
threshold (F = f_bare Q alpha(clay)), and clay is static, so the annual-mean flux
of the switched run is the default annual-mean flux times
aem_alpha(clay, 'legrand') / aem_alpha(clay, 'chappell'), cell by cell. Clay is
the field the run itself used (odem_diag_mean.nc, percent). Totals use
budget_table.annual_total_tg (the one annual-total convention).

Usage:
    python scripts/dev/aem_alpha_rescale.py RUN_DIR [--out FILE.md]
"""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import argparse, os
from pathlib import Path
import numpy as np
import netCDF4 as nc

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from odem import aem_alpha  # noqa: E402
from budget_table import REGIONS, annual_total_tg, region_mask, cell_area  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    with nc.Dataset(run / "odem_annual_mean.nc") as ds:
        lat = np.array(ds["lat"][:]); lon = np.array(ds["lon"][:])
        F = np.nan_to_num(np.array(ds["F_d_mean"][:], dtype=np.float64))
        scheme = getattr(ds, "scheme", "?")
    with nc.Dataset(run / "odem_diag_mean.nc") as ds:
        clay = np.array(ds["clay"][:], dtype=np.float64)
    assert scheme == "aem", f"{run} is not an AEM run ({scheme})"
    a_ch = aem_alpha(clay, "chappell"); a_lg = aem_alpha(clay, "legrand")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(F > 0, a_lg / a_ch, 0.0)
    G = F * ratio
    w = F * cell_area(lat, lon)
    ew = lambda x: float((x[F > 0] * w[F > 0]).sum() / w[F > 0].sum())
    rows = [("Global", annual_total_tg(F, lat, lon), annual_total_tg(G, lat, lon))]
    for r, box in REGIONS.items():
        m = region_mask(lat, lon, box)
        rows.append((r, annual_total_tg(F, lat, lon, m), annual_total_tg(G, lat, lon, m)))
    L = [f"# Pre-registered expectation: AEM with the LeGrand et al. (2023) sandblasting term, from {run.name}", "",
         "F_legrand = F_default x alpha_legrand(clay) / alpha_chappell(clay), cell by cell, clay from the run's "
         "odem_diag_mean.nc (percent); totals by budget_table.annual_total_tg. Script: scripts/dev/aem_alpha_rescale.py.", "",
         f"Emission-weighted alpha over emitting cells: default {ew(a_ch):.4e} m-1, LeGrand {ew(a_lg):.4e} m-1.", "",
         "| region | default Tg/yr | expected LeGrand Tg/yr | ratio |", "|---|---|---|---|"]
    L += [f"| {r} | {f:.1f} | {g:.1f} | {g / f:.4f} |" if f > 0 else f"| {r} | {f:.1f} | {g:.1f} | n/a |"
          for r, f, g in rows]
    L += ["", "Acceptance for the run: every total equal to the expected value with relative difference below 1e-6."]
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
