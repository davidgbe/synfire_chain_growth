#!/bin/bash

## Presets
#SBATCH --job-name=syn_20

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --ntasks=12
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8

## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd synfire_chain_growth/stability_study_triplet_square_fr_21_01_11_b
python run.py --title stdp_scale_20 --stdp_scale 20
deactivate

## Exit
exit 0