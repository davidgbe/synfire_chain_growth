#!/bin/bash
echo $1
caffeinate python3 run.py --title settle_and_drop_${2} --alpha 5e-2 --beta 5e-3 --beta_2 0 --gamma 1e-2 --synfire_prop_dist 1 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --load_run $2 --drop_iter 250