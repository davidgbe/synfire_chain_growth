#!/bin/bash

caffeinate python3 run.py --title shrinking_weights --alpha_1 0 --alpha_2 0 --beta 0 --gamma 0 --fr_single_sym 0 --rng_seed $1 --load_mat $2