#!/bin/bash
echo $1
caffeinate python3 run.py --title pop_stdp_drop_${3} --alpha_1 6e-2 --alpha_2 0.5e-2 --beta 1e-3 --gamma $2 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0.5 --load_run $3