#!/bin/bash
# k14 sensitivity ladder on the MERRA-2 grid, 2006: one full year per held
# factor, run as 12 parallel months each, two factors at a time (each process
# is ~1 GB and mostly numpy). Resumable: finished months are skipped.
# Usage: bash scripts/run_ladder.sh   (from the repository root)
#   ODEM_DATA_ROOT  data directory (required)
#   M2_DIR          MERRA-2 monthly inputs (default $ODEM_DATA_ROOT/merra2_monthly_v2)
#   REGRID_TAG      run-name tag for the statics regridding (default consl)
#   REGRID_ARGS     odem.py regrid options (default: conservative, z0a in log space)
set -eo pipefail
R=${ODEM_DATA_ROOT:?set ODEM_DATA_ROOT to the ODEM data directory}
M2=${M2_DIR:-$R/merra2_monthly_v2}
TAG=${REGRID_TAG:-consl}
RG=${REGRID_ARGS:---regrid conservative --regrid-field z0a=conservative_log}
COMMON="--scheme k14 $RG --soilgrids $R/soilgrids --prigent $R/prigent_data/Prigent_2005_roughness_0.25x0.25.nc --modis-lai $R/modis_lai --landcover $R/modis_landcover/modis_landcover_erodibility_2006.nc"
REF=$R/output_v2/k14_m2_${TAG}_2006
run_one() {
  h=$1
  python scripts/run_year.py --forcing merra2 --meteo-dir $M2 --year 2006 --workers 6 \
      --out ${REF}_hold_$h -- $COMMON --hold $h \
      > $R/output_v2/ladder_${TAG}_$h.log 2>&1 && echo "hold $h done $(date +%H:%M)" || echo "hold $h FAILED"
}
mkdir -p $R/output_v2
for pair in "f_eff_r f_eff_v" "f_moisture C_d" "kappa eta"; do
  set -- $pair
  run_one $1 & run_one $2 & wait
done
python scripts/ladder_table.py --ref $REF \
  f_eff_r=${REF}_hold_f_eff_r f_eff_v=${REF}_hold_f_eff_v \
  f_moisture=${REF}_hold_f_moisture C_d=${REF}_hold_C_d \
  kappa=${REF}_hold_kappa eta=${REF}_hold_eta
