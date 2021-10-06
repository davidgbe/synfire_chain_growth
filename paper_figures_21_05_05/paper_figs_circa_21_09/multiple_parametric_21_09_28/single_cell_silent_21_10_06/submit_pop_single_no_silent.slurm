#!/bin/bash

## Presets
#SBATCH --job-name=pop_single_no_silent

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=00:50:00
#SBATCH --ntasks=12
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
python run.py --title pop_single_no_silent_TITLE --alpha_1 ALPHA_1 --alpha_2 ALPHA_2 --beta 0 --gamma GAMMA --fr_single_sym 0 --rng_seed SEED --dropout_per DROP_SEV --w_e_e_scale_down_factor 0 --w_e_i_scale_down_factor 0 &
wait
deactivate

## Exit
exit 0