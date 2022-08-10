#!/bin/bash

## Presets
#SBATCH --job-name=syn_param

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=14:00:00
#SBATCH --ntasks=10
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd $CURRDIR
python run.py --title secreted_TITLE --index INDEX --rng_seed SEED --hetero_comp_mech COMP_MECH --stdp_type w_minus --dropout_per DROP_SEV  --dropout_iter 1000 --cond CONDITION --w_ee W_E_E --w_ei W_E_I --w_ie W_I_E --silent_fraction PER_SILENT --alpha_5 A5 &
wait
zip -r -qq figures.zip figures
deactivate

## Exit
exit 0
