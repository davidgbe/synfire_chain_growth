#!/bin/bash

caffeinate python3 run.py --title single_cell --alpha 3e-2 --beta 1e-3 --gamma 0 --fr_single_sym True --rng_seed $1