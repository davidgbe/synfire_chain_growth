#!/bin/bash

caffeinate python3 run.py --title strict_i_attractor --alpha_1 3e-2 --alpha_2 3e-3 --beta 1e-3 --gamma 1e-4 --fr_single_sym 0 --rng_seed $1