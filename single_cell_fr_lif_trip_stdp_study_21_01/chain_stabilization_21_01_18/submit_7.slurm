#!/bin/bash

## Presets
#SBATCH --job-name=syn_growth

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --ntasks=12
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8

## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd synfire_chain_growth/single_cell_fr_lif_trip_stdp_study_21_01/chain_stabilization_21_01_18
python run.py --title adapt_test --w_a 8e-4 --fr_scale 2e-5  --fr_penalty 1.5e-3 --stdp_scale 10
deactivate

## Exit
exit 0