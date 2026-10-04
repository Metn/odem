# ODEM v1.1

**Offline Dust Emission Model** for reanalysis-driven global dust source estimation.

ODEM is a standalone Python model that computes global gridded dust emission fluxes from reanalysis meteorological forcing. Its default scheme implements the brittle fragmentation emission parameterization of [Kok et al. (2014)](https://doi.org/10.5194/acp-14-13023-2014) with the parameter choices and implementation of [Leung et al. (2023)](https://doi.org/10.5194/acp-23-6487-2023). Version 1.1 adds the albedo-based drag-partition scheme of Chappell and Webb (2016) and the DEAD scheme as configured in GEOS-Chem, selectable with `--scheme`; the v1.0 defaults are unchanged.

## Features

- Accepts ERA5 or MERRA-2 as meteorological forcing
- Operates at native reanalysis resolution (0.25° for ERA5, 0.5°×0.625° for MERRA-2)
- Single forward pass per timestep — no spin-up, no inter-timestep memory
- Process-based emission physics: soil particle size, moisture inhibition, aerodynamic drag partition, turbulent intermittency
- Three emission scheme families on one common surface state (K14, AEM, DEAD with two source maps)
- Verification suite with 45 physics unit tests and a regression harness against the published runs

## Requirements

- Python 3.10+
- NumPy, SciPy, netCDF4 (the model)
- matplotlib (verification plots and post-processing), cartopy (figure scripts)
- xarray (`scripts/scale_ctune.py`), rasterio (`scripts/brdf_tif_to_nc.py`)

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# ERA5 — single month
python odem.py \
    --era5 era5_data/era5_dust_200601.nc \
    --soilgrids soilgrids/ \
    --prigent prigent_data/Prigent_2005_roughness_0.25x0.25.nc \
    --modis-lai modis_lai/2006 \
    --landcover modis_landcover/modis_landcover_erodibility_2006.nc \
    --output-dir output_2006

# MERRA-2 — single month
python odem.py \
    --merra2 merra2_data/merra2_dust_200601.nc \
    --soilgrids soilgrids/ \
    --prigent prigent_data/Prigent_2005_roughness_0.25x0.25.nc \
    --modis-lai modis_lai/2006 \
    --landcover modis_landcover/modis_landcover_erodibility_2006.nc \
    --output-dir output_2006
```

### Regridding of static inputs

Prigent z0a (0.25 deg), SoilGrids and MODIS land cover (0.05 deg) and MODIS LAI
are put on the meteorology grid by nearest neighbour by default. Add
`--regrid conservative` for a first-order conservative (area-weighted) regrid:
1-D overlap weights in sin(latitude) and longitude (periodic), tensor product,
normalised over valid source cells only, so a cell with any valid source gets
the area mean of its valid part. The default output is unchanged.

### Emission schemes (v1.1)

`--scheme` selects the emission physics; every scheme runs on the same
surface state (land and snow mask, air density, soil moisture, bare-soil
fraction) so that differences between schemes are differences in physics, not
in forcing. `k14` (default) is the Kok et al. (2014) / Leung et al. (2023)
scheme of v1.0; `aem` is the albedo-based drag partition of Chappell and Webb
(2016) with MB95 saltation and needs `--modis-brdf`.

`--aem-sandblasting legrand` is a sensitivity switch for `aem` only: it replaces
the sandblasting term (default: Marticorena and Bergametti 1995 Eq. 47 as coded
in the Google Earth Engine script of Hennen 2022) with Eq. 7 of LeGrand et al.
(2023), and changes nothing else in the chain. The choice is written to the
output attributes as `odem_aem_sandblasting`.

#### DEAD scheme

`dead_g` and `dead_z` implement the DEAD scheme of Zender et al. (2003a) as it
is configured in GEOS-Chem (HEMCO `hcox_dustdead_mod.F`, the version used by
GEOS-Chem 14.4.1 and described in Zhang et al., 2025, Appendix A2): u* from
the 10 m wind by the neutral log profile with z0 = 1e-4 m and no drag
partition, the Iversen and White (1982) dry threshold at 75 um, the Fecan et
al. (1999) moisture factor with the GEOS-Chem threshold w_t = 0.17 + 0.14
f_clay, the White (1979) saltation flux with C = 2.61, the Marticorena and
Bergametti (1995) sandblasting efficiency alpha = 100 x 10^(13.4 f_clay - 6)
m-1 (clay capped at 20 percent), and F_d = Q_s f_bare T S alpha. The two
configurations differ only in the erodibility map S: `dead_g` uses the GOCART /
Ginoux et al. (2001) topographic source function (`GOC_SRC`), `dead_z` the
Zender et al. (2003b) geomorphic erodibility (`EF_GEO`), both read from the
GEOS-Chem `DUST_DEAD` input directory given with `--dead-inputs` and regridded
onto the meteorology grid with the run's `--regrid` method. T is the GEOS-Chem
global mass tuning factor (`--dead-tuning`); its default follows the source
map: 4.7586e-4 with the 2 x 2.5 degree maps (`--dead-s-map 2x25`, the default)
and 4.7376e-4 with the 0.25 degree Ginoux map of Meng et al. (2021)
(`--dead-s-map 025`, `dead_g` only). `--dead-clay-uniform PCT` holds the clay
fraction uniform, as GEOS-Chem does at 20 percent. The 10 m wind (`u10`, `v10`
or MERRA-2 `U10M`, `V10M`) must be in the meteorology file. The docstring of
`flux_dead` lists the GEOS-Chem details that are not reproduced (bare fraction
from `LF_DRY`, fractional snow, bin partition).

```bash
python odem.py     --merra2 merra2_data/merra2_dust_200601.nc     --soilgrids soilgrids/     --prigent prigent_data/Prigent_2005_roughness_0.25x0.25.nc     --modis-lai modis_lai/2006     --landcover modis_landcover/modis_landcover_erodibility_2006.nc     --scheme dead_g --dead-inputs dust_inputs/dead_geoschem_v2019-06     --save-diags --output-dir output_dead_g
```

Other v1.1 flags: `--max-steps N` (development only, stops after N timesteps),
`--modis-brdf` (the `aem` scheme), `--regrid conservative`, `--regrid-field
GROUP=METHOD` (per-static override, methods `nearest`, `conservative`,
`conservative_log`), `--hold FACTOR,...` (K14 factors held at their neutral
values). Every run writes its options as `odem_*` global attributes of the
output files.

### Resolution experiment: ERA5 on the MERRA-2 grid

`regrid_met_conservative.py` regrids every variable of an ERA5 monthly file onto
the MERRA-2 lat/lon grid with the same conservative method and writes a file with
the ERA5 variable and dimension names, so `odem.py --era5` reads it unchanged:

```bash
python regrid_met_conservative.py \
    --src era5_data/era5_dust_200601.nc \
    --target-grid merra2_data/merra2_dust_200601.nc \
    --out era5_on_merra2grid/era5_dust_200601.nc
```

`lsm` becomes a land fraction (thresholded at 0.5 by the model as before); the
global attributes record the method, source, target grid and date.

## Regression harness

`odem_verify.py` checks the physics against external constraints; it does not
check ODEM against itself. `odem_regression.py` does, with two assertions:

1. **Budget.** The global annual total recomputed from each published
   annual-mean file equals the value pinned in `odem_regression_pins.json`
   (relative tolerance 1e-6). The pins cover the six v1.0 runs of the data
   deposit, the four v1.1 primaries on the MERRA-2 grid (K14, AEM, DEAD-G,
   DEAD-Z, 2006) and one development month.
2. **Bitwise.** A freshly computed month is identical, value for value, to the
   archived month it reproduces.

The annual total is the time-mean flux times the spherical cell area times
365.25 days, the convention used everywhere in the model and scripts.

```bash
# 1. Budget check against the data deposit (unpacked into one directory)
python odem_regression.py --check --skip-missing --archive-root path/to/deposit

# 1. Budget check against your own run directories (<root>/<run>/odem_annual_mean.nc)
python odem_regression.py --check --archive-root path/to/runs

# 2. Bitwise check of a month recomputed from the forcing
python odem.py --merra2 merra2_data/merra2_dust_200601.nc ... --output-dir new
python odem_regression.py --compare-month new/odem_200601.nc ref/odem_200601.nc
```

`--archive-root` accepts either layout: a run directory holding
`odem_annual_mean.nc`, or the flat file names of the deposit. `--skip-missing`
reports runs that are not present instead of failing on them, for a deposit
that holds only some of the pinned runs. To reproduce a published year from the
forcing, run it with `scripts/run_year.py` (same options as the pinned run,
recorded in the `odem_*` attributes of its output) and point `--check` at the
new directory. The v1.0 MERRA-2 year is reproduced with `--scheme k14 --regrid
nearest`. `--pin` rewrites the pin file; do so only deliberately, never to make
a failing check pass.

## Input data

All input datasets are freely available from their original providers:

| Dataset | Source | Resolution |
|---------|--------|------------|
| ERA5 | [Copernicus CDS](https://cds.climate.copernicus.eu) | 0.25°, 1-hourly |
| MERRA-2 | [NASA GES DISC](https://disc.gsfc.nasa.gov) | 0.5°×0.625°, 1-hourly |
| SoilGrids v2.0 | [soilgrids.org](https://soilgrids.org) | 250 m |
| MODIS land cover (MCD12C1) | [NASA LP DAAC](https://lpdaac.usgs.gov) | 0.05° |
| MODIS LAI (MOD15A2H) | [NASA LP DAAC](https://lpdaac.usgs.gov) | 500 m |
| Prigent roughness | [Prigent et al. (2005)](https://doi.org/10.1029/2004JD005370) | 0.25° |

See the user manual for detailed variable lists and file naming conventions.

## Repository structure

```
odem.py                  # Main model
odem_analysis.py         # Post-processing (summary, comparison, seasonal)
odem_verify.py           # Physics verification suite (45 unit tests, incl. regrid, DEAD, AEM, merge_year)
odem_regression.py       # Regression harness (pinned budgets, bitwise month comparison)
odem_regression_pins.json  # Pinned global totals of the published runs
regrid_met_conservative.py  # ERA5 -> MERRA-2 grid conservative regrid (resolution experiment)
requirements.txt
scripts/
  run_year.py             # A year as parallel single-month processes (resumable)
  merge_year.py           # Assemble monthly outputs into the files of a single run
  budget_table.py         # Global and regional annual budgets, Kok et al. (2021) ratios, monthly totals
  decompose.py            # Resolution / forcing / convention decomposition of E5/M2
  ladder_table.py, run_ladder.sh  # K14 hold ladder (one factor at its neutral value at a time)
  scale_ctune.py          # Scale output to different C_tune values
  brdf_tif_to_nc.py       # MODIS BRDF shadow parameter GeoTIFF -> NetCDF (AEM input)
  dev/                    # Development checks (bitwise split test, Meng 2021 comparison, DEAD variants)
  figures/               # Paper figure generation (Figs. 2–11)
    gmd_style.py          # Shared matplotlib style
    plot_f02_static_inputs.py
    plot_f03_emission_maps.py
    plot_f04_difference_map.py
    plot_f05_emission_ratio.py
    plot_f06_regional_budget.py
    plot_f07_monthly_timeseries.py
    plot_f09_dustcomm.py
    plot_f10_flux_vs_ustar.py
    plot_f11_ustar_comparison.py
  preprocessing/         # Input data preparation
    prepare_merra2.py     # Merge daily MERRA-2 → monthly NetCDF
    prepare_static_data.py  # Static datasets (clay, landcover, roughness)
```

Figure scripts read data from the path set by the `ODEM_DATA` environment variable:

```bash
export ODEM_DATA=/path/to/dust_model_data
python scripts/figures/plot_f02_static_inputs.py
```

## Reference

Baykara, M.: ODEM v1.1: a standalone dust emission model with interchangeable emission schemes and reanalysis forcings, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-1905, in review, 2026.

## License

MIT
