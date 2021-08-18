#!/bin/bash

caffeinate python3 run.py --title no_stdp_sym --alpha_1 3e-2 --alpha_2 1.5e-2 --beta 0 --gamma 1e-4 --fr_single_sym 1 $1 $2