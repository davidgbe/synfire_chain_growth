#!/bin/bash

## Presets
#SBATCH --job-name=all_rules

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=04:00:00
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
python run.py --title drop --alpha_1 3e-2 --alpha_2 1.5e-2 --beta 0 --gamma 0 --fr_single_sym 0 --rng_seed SEED
deactivate

## Exit
exit 0