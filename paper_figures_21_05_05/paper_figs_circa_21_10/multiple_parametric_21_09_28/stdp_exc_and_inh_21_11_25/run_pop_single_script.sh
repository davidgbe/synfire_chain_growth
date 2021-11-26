#!/bin/bash

caffeinate python3 run.py --title pop_single_small_silent --alpha_1 3e-2 --alpha_2 1e-2 --beta 0 --gamma 0.1e-4 --fr_single_line_attr 2 --rng_seed $1 --dropout_per 0.5 --synfire_prop_dist 1.3