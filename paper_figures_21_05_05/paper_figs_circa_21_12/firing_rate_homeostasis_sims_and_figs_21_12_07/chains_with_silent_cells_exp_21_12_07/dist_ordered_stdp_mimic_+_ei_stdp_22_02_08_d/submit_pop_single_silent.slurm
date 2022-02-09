#!/bin/bash

## Presets
#SBATCH --job-name=pop_single_silent

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=20:00:00
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
python run.py --title moving_fr_setpoints_stdp2_TITLE --alpha ALPHA_1 --beta BETA --beta_2 1e-2 --gamma GAMMA --synfire_prop_dist SYN_PROP_DIST --fr_single_line_attr 1 --rng_seed SEED --dropout_per DROP_SEV --drop_iter 1000 &
wait
deactivate

## Exit
exit 0