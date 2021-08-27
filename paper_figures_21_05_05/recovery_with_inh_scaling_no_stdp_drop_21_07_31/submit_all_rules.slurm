#!/bin/bash

## Presets
#SBATCH --job-name=all_rules

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=07:00:00
#SBATCH --ntasks=12
#SBATCH --mem=80G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8

## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd synfire_chain_growth/paper_figures_21_05_05/recovery_with_inh_scaling_no_stdp_drop_21_07_31
python run.py --title drop --alpha_1 3e-2 --alpha_2 1.5e-2 --beta 0 --gamma 1e-4 --fr_single_sym 0 --rng_seed SEED
deactivate

## Exit
exit 0