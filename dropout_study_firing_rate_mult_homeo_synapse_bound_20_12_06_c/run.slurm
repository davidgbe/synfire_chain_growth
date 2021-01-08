#!/bin/bash

## Presets
#SBATCH --job-name=

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=0:05:00
#SBATCH --ntasks=5
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import

## Commands to run
python3 run.py

## Exit
exit 0