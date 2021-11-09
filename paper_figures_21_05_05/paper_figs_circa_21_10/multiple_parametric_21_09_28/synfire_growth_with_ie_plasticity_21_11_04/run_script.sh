#!/bin/bash

caffeinate python3 run.py --title pop_single_small_silent --alpha_1 0.5 --alpha_2 0.5e-2 --beta 1 --gamma 1e-1 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0.5
# beta 5e-1 2e-1