#!/bin/bash

## Presets
#SBATCH --job-name=single_silent_only

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --ntasks=12
#SBATCH --mem=80G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd $CURRDIR
python run.py --title single_silent_only --alpha_1 3e-2 --alpha_2 1.5e-2 --beta 0 --gamma 0 --fr_single_sym 1 --rng_seed SEED --dropout_per DROP_SEV --w_e_e_scale_down_factor 0.3 --w_e_i_scale_down_factor 0.05
deactivate

## Exit
exit 0