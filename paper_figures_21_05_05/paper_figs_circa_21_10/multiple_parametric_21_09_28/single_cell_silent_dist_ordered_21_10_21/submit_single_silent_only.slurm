#!/bin/bash

## Presets
#SBATCH --job-name=single_silent_only

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
python run.py --title single_silent_only_TITLE --alpha_1 ALPHA_1 --alpha_2 ALPHA_2 --beta 0 --gamma 0 --fr_single_sym 1 --rng_seed SEED --dropout_per DROP_SEV &
wait
deactivate

## Exit
exit 0