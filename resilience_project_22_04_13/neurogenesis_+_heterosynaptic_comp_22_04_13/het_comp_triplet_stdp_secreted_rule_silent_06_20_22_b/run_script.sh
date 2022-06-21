#!/bin/bash

caffeinate python3 run.py --title summed_bound --index 0 --rng_seed $3 --hetero_comp_mech secreted_regulation --stdp_type w_minus --dropout_per $1 --dropout_iter $2 --cond all --w_ee 1.2e-3 --w_ei 7e-5 --w_ie 6e-5 --silent_fraction $4