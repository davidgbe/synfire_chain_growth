#!/bin/bash

caffeinate python3 run.py --title single_cell --alpha_1 3e-2 --alpha_2 0.5e-3 --beta 1e-3 --gamma 0 --fr_single_sym 1 --rng_seed $1