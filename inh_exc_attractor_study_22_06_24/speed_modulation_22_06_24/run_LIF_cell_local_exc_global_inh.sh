#!/bin/bash

caffeinate python3 run_LIF_cell_local_exc_global_inh.py --title LIF_cell_local_exc_global_inh --index 0 --rng_seed $3  --dropout_per $1 --dropout_iter $2 --w_ee 5e-4 --beta 0 --w_ei 10e-4 --w_ie 4e-5 --w_u 0