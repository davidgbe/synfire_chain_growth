#!/bin/bash

caffeinate python3 run.py --title pop_single_small_silent --alpha_1 5e-2 --alpha_2 5e-1 --beta 0 --gamma 0.5e-2 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --synfire_prop_dist 1.3