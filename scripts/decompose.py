#!/usr/bin/env python
"""Decompose an inter-reanalysis emission difference into resolution,
forcing and convention terms, in log space.

For one scheme, given the annual budgets of
  E5   ERA5 forcing on its native 0.25 deg grid (3-hourly instantaneous),
  E5c  ERA5 forcing conservatively regridded to the MERRA-2 grid before the
       emission calculation,
  M2   MERRA-2 forcing on its native grid (hourly-mean winds or u*),
  M2i  MERRA-2 forcing with instantaneous hourly winds (M2I1NXASM; wind-driven
       schemes only, MERRA-2 has no instantaneous u*),
the split is exact in log space (emission is multiplicative in its factors):

  wind-driven (M2i given):
    ln(E5/M2i) = ln(E5/E5c) + ln(E5c/M2i)
                 resolution   forcing at matched grid and matched convention
    ln(M2i/M2)   convention (instantaneous over hourly mean)
    E5c/M2 = (E5c/M2i)(M2i/M2) is printed as a reference column: forcing at
    matched grid, convention bounded (not matched).
  u*-driven (no M2i, K14):
    ln(E5/M2) = ln(E5/E5c) + ln(E5c/M2)
    where E5c/M2 carries the unremovable convention gap. With --convention-bound
    b (fraction, e.g. 0.10 from the 6 h smoothing arm), a convention-matched
    MERRA-2 year lies below M2/(1-b), so the matched-convention forcing term lies
    in [(1-b) E5c/M2, E5c/M2]; printed for the global row only (the bound is global).

Global and the seven paper regions.

Usage:
  python scripts/decompose.py --e5 <dir> --e5c <dir> --m2 <dir> [--m2i <dir>]
                              [--convention-bound 0.10] [--label k14]
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
    p.add_argument('--convention-bound', type=float, default=None,
                   help='upper bound on the u* convention gap as a fraction (K14; AC1: 0.10)')
    p.add_argument('--label', default='')
    a = p.parse_args()
    runs = {'E5': a.e5, 'E5c': a.e5c, 'M2': a.m2}
    if a.m2i:
        runs['M2i'] = a.m2i
    B = {}
    for k, d in runs.items():
        B[k], meta, _ = budgets(os.path.join(d, 'odem_annual_mean.nc'))
        print(f'{k:4s} {meta.get("scheme", ""):6s} {B[k]["Global"]:9.1f} Tg/yr  {os.path.basename(d.rstrip("/"))}')
    regions = ['Global'] + list(REGIONS)
    if a.m2i:
        hdr = ('| region | resolution E5/E5c | forcing, matched conv. E5c/M2i | convention M2i/M2 '
               '| forcing, bounded E5c/M2 (reference) | total E5/M2i |')
    else:
        hdr = '| region | resolution E5/E5c | forcing E5c/M2 (convention bounded) | total E5/M2 |'
    print(f'\nScheme {a.label}: multiplicative factors (log-space additive)')
    print(hdr)
    print('|' + '---|' * (hdr.count('|') - 1))
    for r in regions:
        e5, e5c, m2 = B['E5'][r], B['E5c'][r], B['M2'][r]
        vals = [e5, e5c, m2] + ([B['M2i'][r]] if a.m2i else [])
        if min(vals) <= 0:
            print(f'| {r} | n/a |'); continue
        if a.m2i:
            m2i = B['M2i'][r]
            row = [r, f'{e5 / e5c:.3f}', f'{e5c / m2i:.3f}', f'{m2i / m2:.3f}', f'{e5c / m2:.3f}', f'{e5 / m2i:.3f}']
        else:
            row = [r, f'{e5 / e5c:.3f}', f'{e5c / m2:.3f}', f'{e5 / m2:.3f}']
        print('| ' + ' | '.join(row) + ' |')
    e5, e5c, m2 = B['E5']['Global'], B['E5c']['Global'], B['M2']['Global']
    if a.m2i:
        m2i = B['M2i']['Global']
        tot = math.log(e5 / m2i)
        print(f'\nGlobal: ln(E5/M2i) = {tot:+.4f} = resolution {math.log(e5 / e5c):+.4f} '
              f'+ forcing (matched conv.) {math.log(e5c / m2i):+.4f}; '
              f'convention ln(M2i/M2) = {math.log(m2i / m2):+.4f}')
    else:
        tot = math.log(e5 / m2)
        print(f'\nGlobal: ln(E5/M2)= {tot:+.4f} = resolution {math.log(e5 / e5c):+.4f} '
              f'+ forcing (convention bounded) {math.log(e5c / m2):+.4f}')
        if a.convention_bound is not None:
            b = a.convention_bound
            print(f'        convention gap bounded above by {100 * b:.0f} %: matched-convention MERRA-2 '
                  f'< {m2 / (1 - b):.1f} Tg/yr, so the matched-convention forcing term E5c/M2 lies in '
                  f'[{(1 - b) * e5c / m2:.3f}, {e5c / m2:.3f}]')


if __name__ == '__main__':
    main()
