#!/usr/bin/env python
"""Sensitivity ladder table for the k14 scheme: the ratio of each --hold run's
annual budget to the reference run, global and per paper region, so that the
'dominant departing factor' of the headline table can be read off directly.

Usage:
  python scripts/ladder_table.py --ref <run_dir> f_eff_r=<dir> f_eff_v=<dir> f_moisture=<dir> C_d=<dir> kappa=<dir> eta=<dir>
Each hold run is label=directory. Ratios above 1 mean the factor suppresses
emission in the reference run by that amount.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from budget_table import budgets, REGIONS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True)
    p.add_argument('holds', nargs='+', help='label=run_dir for each held factor')
    a = p.parse_args()
    ref, meta, _ = budgets(os.path.join(a.ref, 'odem_annual_mean.nc'))
    print(f'reference: {meta.get("source", "")}  global {ref["Global"]:.1f} Tg/yr')
    regions = ['Global'] + list(REGIONS)
    rows = []
    for spec in a.holds:
        label, d = spec.split('=', 1)
        b, m, _ = budgets(os.path.join(d, 'odem_annual_mean.nc'))
        rows.append((label, b, m.get('source', '')))
    print('| held factor | ' + ' | '.join(regions) + ' |')
    print('|' + '---|' * (1 + len(regions)))
    for label, b, src in rows:
        print('| ' + label + ' | ' + ' | '.join(f'{b[r] / ref[r]:.3f}' if ref[r] > 0 else 'n/a' for r in regions) + ' |')
    print('\nDominant departing factor per region (largest |ln ratio|):')
    import math
    for r in regions:
        best = max(rows, key=lambda t: abs(math.log(t[1][r] / ref[r])) if ref[r] > 0 and t[1][r] > 0 else 0)
        print(f'  {r:12s} {best[0]:12s} x{best[1][r] / ref[r]:.2f}')


if __name__ == '__main__':
    main()
