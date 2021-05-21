#!/bin/bash

## Presets
#SBATCH --job-name=single

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=6:40:00
#SBATCH --ntasks=12
#SBATCH --mem=80G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8

## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd synfire_chain_growth/paper_figures_21_05_05/recovery_with_plasticity_rules_comp_21_05_13
python run.py --title single_cell --alpha 3e-2 --beta 1e-3 --gamma 0 --fr_single_sym True --rng_seed SEED
deactivate

## Exit
exit 0