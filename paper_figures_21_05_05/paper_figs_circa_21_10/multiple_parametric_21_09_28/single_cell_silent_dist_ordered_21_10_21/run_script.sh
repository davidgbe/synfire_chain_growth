#!/bin/bash

caffeinate python3 run.py --title single_only --alpha_1 3e-2 --alpha_2 0.5e-2 --beta 0 --gamma 0 --fr_single_sym 1 --rng_seed $1 --dropout_per 0.5