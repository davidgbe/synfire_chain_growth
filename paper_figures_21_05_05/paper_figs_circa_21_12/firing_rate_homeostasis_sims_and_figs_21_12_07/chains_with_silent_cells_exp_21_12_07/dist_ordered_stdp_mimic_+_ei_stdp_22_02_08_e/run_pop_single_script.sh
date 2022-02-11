#!/bin/bash

caffeinate python3 run.py --title stdp --alpha 5e-2 --beta 5e-3 --beta_2 0 --gamma 1e-2 --fr_single_line_attr 1 --rng_seed $1 --dropout_per 0.5 --synfire_prop_dist 1. --drop_iter $2