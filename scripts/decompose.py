#!/usr/bin/env python
"""Decompose an inter-reanalysis emission difference into resolution,
forcing-at-matched-grid and sampling-convention terms, in log space.

For one scheme, given the annual budgets of
  E5   ERA5 forcing on its native 0.25 deg grid,
  E5c  ERA5 forcing conservatively regridded to the MERRA-2 grid before the
       emission calculation,
  M2   MERRA-2 forcing on its native grid (time-mean hourly winds or u*),
  M2i  MERRA-2 forcing with instantaneous hourly winds (wind-driven schemes),
the confounded ratio E5/M2 splits exactly as
  ln(E5/M2) = ln(E5/E5c) + ln(E5c/M2)            [resolution + forcing]
and, when M2i exists,
  ln(E5/M2i) = ln(E5/E5c) + ln(E5c/M2) + ln(M2/M2i) [+ sampling convention]
so each term is a factor and the factors multiply to the total. Emission is
multiplicative in its factors, which is why the split is done in log space.
Global and the seven paper regions.

Usage:
  python scripts/decompose.py --e5 <dir> --e5c <dir> --m2 <dir> [--m2i <dir>] [--label k14]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from budget_table import budgets, REGIONS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--e5', required=True); p.add_argument('--e5c', required=True)
    p.add_argument('--m2', required=True); p.add_argument('--m2i', default=None)
    p.add_argument('--label', default='')
    a = p.parse_args()
    runs = {'E5': a.e5, 'E5c': a.e5c, 'M2': a.m2}
    if a.m2i:
        runs['M2i'] = a.m2i
    B = {}
    for k, d in runs.items():
        B[k], meta, _ = budgets(os.path.join(d, 'odem_annual_mean.nc'))
        print(f'{k:4s} {meta.get("scheme", ""):6s} {B[k]["Global"]:9.1f} Tg/yr  {d}')
    regions = ['Global'] + list(REGIONS)
    hdr = '| region | E5/M2 total | resolution E5/E5c | forcing E5c/M2 |'
    if a.m2i:
        hdr += ' sampling M2/M2i | E5/M2i total |'
    print(f'\nScheme {a.label}: multiplicative factors (log-space additive)')
    print(hdr)
    print('|' + '---|' * (hdr.count('|') - 1))
    for r in regions:
        e5, e5c, m2 = B['E5'][r], B['E5c'][r], B['M2'][r]
        if min(e5, e5c, m2) <= 0:
            print(f'| {r} | n/a |'); continue
        row = [r, f'{e5 / m2:.3f}', f'{e5 / e5c:.3f}', f'{e5c / m2:.3f}']
        if a.m2i:
            m2i = B['M2i'][r]
            row += [f'{m2 / m2i:.3f}', f'{e5 / m2i:.3f}']
        print('| ' + ' | '.join(row) + ' |')
    # share of the log difference carried by each term, global
    e5, e5c, m2 = B['E5']['Global'], B['E5c']['Global'], B['M2']['Global']
    tot = math.log(e5 / m2)
    if tot != 0:
        print(f'\nGlobal: ln(E5/M2) = {tot:+.4f} = resolution {math.log(e5 / e5c):+.4f} '
              f'({100 * math.log(e5 / e5c) / tot:.0f} %) + forcing {math.log(e5c / m2):+.4f} '
              f'({100 * math.log(e5c / m2) / tot:.0f} %)')
    if a.m2i:
        m2i = B['M2i']['Global']
        print(f'        sampling term ln(M2/M2i) = {math.log(m2 / m2i):+.4f}; '
              f'ln(E5/M2i) = {math.log(e5 / m2i):+.4f}')


if __name__ == '__main__':
    main()
