#!/bin/bash

caffeinate python3 run.py --title pop_silent --alpha_1 0 --alpha_2 1.5e-2 --beta 0 --gamma 1e-4 --fr_single_sym 0 --rng_seed $1 --dropout_per 0.5 --w_e_e_scale_down_factor 0.3 --w_e_i_scale_down_factor 0.05