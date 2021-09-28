#!/bin/bash

caffeinate python3 run.py --title single_only --alpha_1 3e-2 --alpha_2 1.5e-2 --beta 0 --gamma 0 --fr_single_sym 1 --rng_seed $1 --dropout_per 0.5 --w_e_e_scale_down_factor 0 --w_e_i_scale_down_factor 0