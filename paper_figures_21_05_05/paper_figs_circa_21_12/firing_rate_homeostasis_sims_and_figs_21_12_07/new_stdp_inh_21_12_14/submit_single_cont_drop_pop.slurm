#!/bin/bash

## Presets
#SBATCH --job-name=single_silent_only

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=10:00:00
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
python run.py --title pop_stdp_drop_3_TITLE --alpha_1 6e-2 --alpha_2 0.5e-2 --beta 1e-2 --gamma GAMMA --fr_single_line_attr 1 --rng_seed SEED --dropout_per 0.5 --load_run LOADED_RUN_NAME --drop_iter 80 &
wait
deactivate

## Exit
exit 0