#!/bin/bash

caffeinate python3 run.py --title loaded --index 0 --rng_seed $5 --hetero_comp_mech secreted_regulation --stdp_type w_minus --dropout_per $3 --dropout_iter $4 --cond no_repl_no_syn --w_ee 1.2e-3 --w_ei 7e-5 --w_ie 6e-5 --load_run $1 $2