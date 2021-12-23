#!/bin/bash
echo $1
caffeinate python3 run.py --title pop_stdp_drop_wide_attr_${2} --alpha_1 6e-2 --alpha_2 0.5e-2 --beta 1e-2 --gamma 1e-4 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --load_run $2 --synfire_prop_dist 1.3 --drop_iter 50