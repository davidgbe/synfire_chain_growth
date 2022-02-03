#!/bin/bash

caffeinate python3 run.py --title ei_stdp_10ms --alpha 3e-2 --beta 2e-3 --gamma 1e-2 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --synfire_prop_dist 1. --drop_iter $2