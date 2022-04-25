#!/bin/bash

caffeinate python3 run.py --title weight_tracked --index 0 --rng_seed $3 --hetero_comp_mech $1 --stdp_type w_minus --dropout_per $2 --w_ee 1.8e-3 --w_ei 5e-5 --w_ie 4.5e-5