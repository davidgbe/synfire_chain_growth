#!/bin/bash

caffeinate python3 run.py --title summed_bound --index 0 --rng_seed $3 --hetero_comp_mech secreted_regulation --stdp_type w_minus --dropout_per $1 --dropout_iter $2 --cond no_repl_no_syn --w_ee 1.4e-3 --w_ei 10e-5 --w_ie 2e-5 --silent_fraction $4 --alpha_5 1.