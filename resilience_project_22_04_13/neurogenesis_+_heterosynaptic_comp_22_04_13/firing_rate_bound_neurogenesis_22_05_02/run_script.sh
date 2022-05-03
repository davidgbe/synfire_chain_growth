#!/bin/bash

caffeinate python3 run.py --title summed_bound --index 0 --rng_seed $4 --hetero_comp_mech $1 --stdp_type w_minus --dropout_per $2 --cond $3 --w_ee 2.4e-3 --w_ei 5e-5 --w_ie 4.5e-5