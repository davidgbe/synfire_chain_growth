#!/bin/bash

## Presets
#SBATCH --job-name=pop_single_silent

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=02:30:00
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
python run.py --title pop_single_small_silent_TITLE --alpha_1 ALPHA_1 --alpha_2 ALPHA_2 --beta 0 --gamma GAMMA --synfire_prop_dist SYN_PROP_DIST --fr_single_line_attr 2 --rng_seed SEED --dropout_per DROP_SEV &
wait
deactivate

## Exit
exit 0