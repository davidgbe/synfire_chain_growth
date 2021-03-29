#!/bin/bash

## Presets
#SBATCH --job-name=syn_growth

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
cd synfire_chain_growth/timing_recovery_exps_21_01_19/LIF_neurons_structed_inh_timed_21_03_23_b
python run.py --title stability_test --w_a 0 --fr_penalty 0.5 --stdp_scale 10 --beta 2.5
deactivate

## Exit
exit 0