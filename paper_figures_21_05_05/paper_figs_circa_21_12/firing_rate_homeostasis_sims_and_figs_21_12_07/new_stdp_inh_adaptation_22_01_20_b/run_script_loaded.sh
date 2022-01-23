#!/bin/bash

caffeinate python3 run.py --title loaded --alpha_1 3e-2 --alpha_2 0.5e-2 --beta 1e-1 --gamma 0 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0.5 --load_run single_only_ff_0.5_eir_0.7_ier_0.9_2021-11-28--12:54--16:5880