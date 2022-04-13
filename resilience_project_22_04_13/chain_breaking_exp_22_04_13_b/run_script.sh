#!/bin/bash

caffeinate python3 run.py --title uniform_short_inh_delay_dp_$2 --index 0 --rng_seed $1 --dropout_per $2 --w_ee 0.8e-3 --w_ei 3e-5 --w_ie 3e-5