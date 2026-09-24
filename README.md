# ODEM v1.0

**Offline Dust Emission Model** for reanalysis-driven global dust source estimation.

ODEM is a standalone Python model that computes global gridded dust emission fluxes from reanalysis meteorological forcing. It implements the brittle fragmentation emission parameterization of [Kok et al. (2014)](https://doi.org/10.5194/acp-14-13023-2014) with the parameter choices and implementation of [Leung et al. (2023)](https://doi.org/10.5194/acp-23-6487-2023).

## Features

- Accepts ERA5 or MERRA-2 as meteorological forcing
- Operates at native reanalysis resolution (0.25° for ERA5, 0.5°×0.625° for MERRA-2)
- Single forward pass per timestep — no spin-up, no inter-timestep memory
- Process-based emission physics: soil particle size, moisture inhibition, aerodynamic drag partition, turbulent intermittency
- Verification suite with 38 physics unit tests

## Requirements

- Python 3.10+
- NumPy, SciPy, xarray, netCDF4, pandas
- matplotlib (post-processing only)

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
odem_verify.py           # Physics verification suite (38 tests) + conservative-regrid unit tests
regrid_met_conservative.py  # ERA5 -> MERRA-2 grid conservative regrid (resolution experiment)
requirements.txt
scripts/
  scale_ctune.py          # Scale output to different C_tune values
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

Baykara, M.: ODEM v1.0: an offline dust emission model for reanalysis-driven source estimation, Geosci. Model Dev., in preparation, 2026.

## License

MIT
