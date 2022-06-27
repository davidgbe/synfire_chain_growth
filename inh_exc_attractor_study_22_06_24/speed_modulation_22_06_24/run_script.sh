#!/bin/bash

caffeinate python3 run.py --title inh_exc --index 0 --rng_seed $3  --dropout_per $1 --dropout_iter $2 --w_ee 6e-5 --beta 5 --w_ei 7e-4 --w_ie 6e-4 --w_u 1e-4