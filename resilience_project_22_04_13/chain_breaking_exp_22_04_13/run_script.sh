#!/bin/bash

caffeinate python3 run.py --title uniform_short_inh_delay_w_e_i_7e-5_dp_$2 --index 0 --rng_seed $1 --dropout_per $2 --w_ee 2e-3 --w_ei 7e-5 --w_ie 7.5e-5