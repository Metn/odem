#!/bin/bash
# k14 sensitivity ladder on the MERRA-2 grid, 2006, conservative statics:
# one full year per held factor, run as 12 parallel months each, two factors
# at a time (24 processes on a 16-thread desktop is fine: each process is
# ~1 GB and mostly numpy). Resumable: finished months are skipped.
# Usage: bash scripts/run_ladder.sh   (from D:/coding_folder/Python/odem)
set -eo pipefail
R=ODEM_DATA_ROOT
COMMON="--scheme k14 --regrid conservative --soilgrids $R/soilgrids --prigent $R/prigent_data/Prigent_2005_roughness_0.25x0.25.nc --modis-lai $R/modis_lai --landcover $R/modis_landcover/modis_landcover_erodibility_2006.nc"
run_one() {
  h=$1
  python scripts/run_year.py --forcing merra2 --meteo-dir $R/merra2_monthly_v2 --year 2006 --workers 6 \
      --out $R/output_v2/k14_m2_cons_2006_hold_$h -- $COMMON --hold $h \
      > $R/output_v2/ladder_$h.log 2>&1 && echo "hold $h done $(date +%H:%M)" || echo "hold $h FAILED"
}
mkdir -p $R/output_v2
for pair in "f_eff_r f_eff_v" "f_moisture C_d" "kappa eta"; do
  set -- $pair
  run_one $1 & run_one $2 & wait
done
python scripts/ladder_table.py --ref $R/output_v2/k14_m2_cons_2006 \
  f_eff_r=$R/output_v2/k14_m2_cons_2006_hold_f_eff_r f_eff_v=$R/output_v2/k14_m2_cons_2006_hold_f_eff_v \
  f_moisture=$R/output_v2/k14_m2_cons_2006_hold_f_moisture C_d=$R/output_v2/k14_m2_cons_2006_hold_C_d \
  kappa=$R/output_v2/k14_m2_cons_2006_hold_kappa eta=$R/output_v2/k14_m2_cons_2006_hold_eta
