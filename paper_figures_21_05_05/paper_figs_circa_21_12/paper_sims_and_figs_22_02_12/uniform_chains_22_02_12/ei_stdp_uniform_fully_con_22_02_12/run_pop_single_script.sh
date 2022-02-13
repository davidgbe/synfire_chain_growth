#!/bin/bash

caffeinate python3 run.py --title settle --alpha 5e-2 --beta 5e-3 --gamma 0 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --drop_iter 10000