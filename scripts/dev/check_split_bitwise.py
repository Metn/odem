"""Old (v1.0.1, e2b54e8) vs new (split) compute_emission on real timesteps: bitwise.

Usage: python scripts/dev/check_split_bitwise.py [odem_v1.py] [nsteps]
Without odem_v1.py the v1.0.1 module is extracted from git (commit e2b54e8).
"""
import os, sys, subprocess, tempfile, importlib.util, pathlib, numpy as np
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import odem as new

V1_COMMIT = 'e2b54e8'
if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
    v1_path = sys.argv[1]
    argn = sys.argv[2:]
else:
    src = subprocess.run(['git', '-C', str(REPO), 'show', f'{V1_COMMIT}:odem.py'],
                         check=True, capture_output=True).stdout
    tmp = tempfile.NamedTemporaryFile(suffix='_odem_v1.py', delete=False)
    tmp.write(src); tmp.close()
    v1_path = tmp.name
    argn = sys.argv[1:]
spec = importlib.util.spec_from_file_location('odem_v1', v1_path)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

R = os.environ.get('ODEM_DATA_ROOT')
if not R:
    raise SystemExit('set ODEM_DATA_ROOT to the ODEM data directory (soilgrids/, prigent_data/, modis_lai/, ...)')
cases = [('merra2', f'{R}/merra2_monthly/merra2_dust_200601.nc'), ('era5', f'{R}/era5/era5_dust_200601.nc')]
nsteps = int(argn[0]) if argn else 6
for src, path in cases:
    inp = new.InputData(path, f'{R}/soilgrids', f'{R}/prigent_data/Prigent_2005_roughness_0.25x0.25.nc',
                        f'{R}/modis_lai', landcover_file=f'{R}/modis_landcover/modis_landcover_erodibility_2006.nc',
                        meteo_source=src)
    inp.load()
    lai0 = inp.get_modis_lai(inp.times[0])
    D_p = new.median_particle_diameter(inp.clay, inp.silt, lai0)
    static = {'lsm': inp.lsm, 'z0a': inp.z0a, 'clay': inp.clay, 'silt': inp.silt, 'bulk_den': inp.bulk_den,
              'D_p': D_p, 'z0s': D_p / 15.0, 'f_erodible': inp.f_erodible}
    idx = np.linspace(0, inp.nt - 1, nsteps).astype(int)
    bad = 0
    for t in idx:
        met = inp.get_timestep(int(t)); lai = inp.get_modis_lai(inp.times[t])
        Fo, do = old.compute_emission(met, static, lai); Fn, dn = new.compute_emission(met, static, lai)
        ok = np.array_equal(Fo, Fn, equal_nan=True)
        dbad = [k for k in do if not np.array_equal(np.asarray(do[k]), np.asarray(dn[k]), equal_nan=True)]
        extra = set(dn) - set(do)
        print(f'{src} t={int(t):4d} F_d bitwise={ok} sum={Fn.sum():.6e} diag_mismatch={dbad} new_keys={sorted(extra)}')
        bad += (not ok) + len(dbad)
    print(f'== {src}: {"PASS" if bad == 0 else "FAIL"} over {len(idx)} timesteps')
