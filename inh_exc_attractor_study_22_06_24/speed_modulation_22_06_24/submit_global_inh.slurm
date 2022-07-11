#!/bin/bash

## Presets
#SBATCH --job-name=syn_param

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=1:10:00
#SBATCH --ntasks=10
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd /mmfs1/gscratch/labfairhall/davidgbe
source davidgbe-env/bin/activate
cd $CURRDIR
python run_LIF_cell_local_exc_global_inh.py --title TITLE --index INDEX --rng_seed SEED --dropout_per DROP_SEV --dropout_iter 5 --w_ee W_E_E_F --beta BETA --w_ei W_E_I --w_ie W_I_E --w_u W_U &
wait
# zip -r -qq figures.zip figures
deactivate

## Exit
exit 0
