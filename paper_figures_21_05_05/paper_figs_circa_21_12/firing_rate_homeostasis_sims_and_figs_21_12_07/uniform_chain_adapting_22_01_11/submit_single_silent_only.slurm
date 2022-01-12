#!/bin/bash

## Presets
#SBATCH --job-name=single_silent_only

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=24:00:00
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
python run.py --title tau_ei_stdp_10ms_TITLE --alpha_1 6e-2 --alpha_2 0.5e-2 --beta BETA --gamma 0 --fr_single_line_attr 0 --rng_seed SEED --dropout_per 0.5 --tau_stdp_ei T_S_EI --t_r_i TRI --stdp_dep STDP_EI_DEP --drop_iter 10000 &
wait
deactivate

## Exit
exit 0