#!/bin/bash

caffeinate python3 run.py --title $1 --index 0 --rng_seed $3 --hetero_comp_mech $1 --stdp_type w_minus --dropout_per $2 --w_ee 0.8e-3 --w_ei 2.5e-5 --w_ie 9e-5