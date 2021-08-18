#!/bin/bash

caffeinate python3 run.py --title fr_modulated_stdp --alpha_1 3e-2 --alpha_2 1.5e-2 --beta 1e-2 --gamma 1e-4 --fr_single_sym 0 $1 $2