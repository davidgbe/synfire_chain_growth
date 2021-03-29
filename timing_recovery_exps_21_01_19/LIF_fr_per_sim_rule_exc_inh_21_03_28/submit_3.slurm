#!/bin/bash

## Presets
#SBATCH --job-name=fr_recovery

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --ntasks=12
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8

## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd synfire_chain_growth/timing_recovery_exps_21_01_19/LIF_fr_per_sim_rule_21_03_28_b
caffeinate python run.py --w_a 0 --p_ff 0.2 --p_r 0.1
deactivate

## Exit
exit 0