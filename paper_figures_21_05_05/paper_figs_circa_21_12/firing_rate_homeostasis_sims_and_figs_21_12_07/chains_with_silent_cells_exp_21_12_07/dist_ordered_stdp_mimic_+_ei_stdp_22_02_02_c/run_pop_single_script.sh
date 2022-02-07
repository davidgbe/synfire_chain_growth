#!/bin/bash

caffeinate python3 run.py --title stdp --alpha 3e-2 --beta 2e-3 --beta_2 1e-2 --gamma 1e-1 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --synfire_prop_dist 0.5 --drop_iter $2