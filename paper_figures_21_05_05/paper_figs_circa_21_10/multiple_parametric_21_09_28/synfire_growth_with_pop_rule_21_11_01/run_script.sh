#!/bin/bash

caffeinate python3 run.py --title pop_single_small_silent --alpha_1 1 --alpha_2 0.5e-2 --beta 1 --gamma 5e-2 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0.5