"""Dev-only: run odem.py with flux_dead keyword overrides that are not on the CLI.

Usage: python scripts/dev/run_dead_variant.py --clay-pct 20 -- <odem.py args>
"""
import sys, functools, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import odem

argv = sys.argv[1:]
split = argv.index('--')
opts, rest = argv[:split], argv[split + 1:]
kw = {}
if '--clay-pct' in opts:
    kw['clay_pct'] = float(opts[opts.index('--clay-pct') + 1])
for key in ('dead_g', 'dead_z'):
    odem.SCHEMES[key] = functools.partial(odem.flux_dead, **kw)
print(f"flux_dead overrides: {kw}", flush=True)
sys.argv = ['odem.py'] + rest
odem.main()
