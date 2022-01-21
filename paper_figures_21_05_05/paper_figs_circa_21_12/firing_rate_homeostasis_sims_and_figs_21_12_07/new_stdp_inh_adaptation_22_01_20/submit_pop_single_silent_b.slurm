#!/bin/bash

## Presets
#SBATCH --job-name=pop_single_adapting

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=15:00:00
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
python run.py --title settledrop_TITLE --alpha_1 ALPHA_1 --alpha_2 ALPHA_2 --beta BETA --gamma GAMMA --fr_single_line_attr FR_LINE_ATTR --rng_seed SEED --dropout_per DROP_SEV --drop_iter 2000 &
wait
deactivate

## Exit
exit 0