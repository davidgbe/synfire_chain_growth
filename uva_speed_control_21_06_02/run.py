from copy import deepcopy as copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
import pickle
from collections import OrderedDict
import os
from scipy.ndimage.interpolation import shift
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.sparse import csc_matrix, csr_matrix, kron
from functools import reduce
import argparse

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--rng_seed', metavar='r', type=int, nargs=1)

args = parser.parse_args()

print(args)

# PARAMS
## NEURON AND NETWORK MODEL
M = Generic(
    # Excitatory membrane
    C_M_E=1e-6,  # membrane capacitance
    G_L_E=.4e-3,  # membrane leak conductance (T_M (s) = C_M (F/cm^2) / G_L (S/cm^2))
    E_L_E=-.067,  # membrane leak potential (V)
    V_TH_E=-.043,  # membrane spike threshold (V)
    T_R_E=12e-3,  # refractory period (s)
    E_R_E=-0.067, # reset voltage (V)
    
    # Inhibitory membrane
    C_M_I=1e-6,
    G_L_I=.4e-3, 
    E_L_I=-.057,
    V_TH_I=-.043,
    T_R_I=1e-3,
    E_R_I=-.057, # reset voltage (V)
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.09, E_A=-.07, T_E=.004, T_I=.004, T_A=.006,
    
    N_EXC=2000,
    N_SILENT=0,
    N_INH=450,
    M=20,
    
    # Input params
    DRIVING_HZ=2, # 2 Hz lambda Poisson input to system
    N_DRIVING_CELLS=20,
    PROJECTION_NUM=20,
    INPUT_STD=1e-3,
    BURST_T=1.5e-3,
    
    # OTHER INPUTS
    SGM_N=.5e-10,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input

    # Connection probabilities
    CON_PROB_FF=0.7,
    CON_PROB_R=0.,
    E_I_CON_PER_LINK=1,
    I_E_CON_PROB=0.8,

    # Weights
    W_E_I_R=2e-5,
    W_I_E_R=0.3e-5,
    W_A=0,
    W_INITIAL=0.26 * 0.004 * 1.6,

    F_B=500,
    T_B=10e-3,
    SIGMA_B=0.5e-3,
)

S = Generic(RNG_SEED=args.rng_seed[0], DT=0.2e-3, T=400e-3, EPOCHS=2000)
np.random.seed(S.RNG_SEED)

M.RAND_WEIGHT_MAX = M.W_INITIAL / (M.M * M.N_EXC)
M.W_U_E = M.W_INITIAL / M.PROJECTION_NUM * 2

## SMLN

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)


def generate_ff_chain(size, unit_size, unit_funcs, ff_deg=[0, 1], tempering=[1., 1.]):
    if size % unit_size != 0:
        raise ValueError('unit_size does not evenly divide size')

    if len(ff_deg) != len(tempering):
        raise ValueError('ff_deg and tempering must be the same length')

    n_units = int(size / unit_size)
    chain_order = np.arange(0, n_units, dtype=int)
    mat = np.zeros((size, size))

    for idx in chain_order:
        layer_start = unit_size * idx
        for j, ff_idx in enumerate(ff_deg):
            if idx + ff_idx < len(chain_order) and idx + ff_idx >= 0:
                ff_layer_idx = chain_order[idx + ff_idx]
                ff_layer_start = unit_size * ff_layer_idx

                mat[ff_layer_start : ff_layer_start + unit_size, layer_start : layer_start + unit_size] = unit_funcs[j]() * tempering[j]
    return mat

def generate_exc_ff_chain(m):
    def rec_unit_func():
        return mat_1_if_under_val(m.CON_PROB_R, (m.PROJECTION_NUM, m.PROJECTION_NUM))

    def ff_unit_func():
        w = m.W_INITIAL / m.PROJECTION_NUM
        return gaussian_if_under_val(m.CON_PROB_FF, (m.PROJECTION_NUM, m.PROJECTION_NUM), w, 0.5 * w)

    unit_funcs = [rec_unit_func, ff_unit_func]

    return generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, unit_funcs)

def generate_local_con(m, ff_deg=[0, 1, 2]):
    def unit_func():
        return np.random.rand(m.PROJECTION_NUM, m.PROJECTION_NUM)

    unit_funcs = [unit_func] * len(ff_deg)

    return m.RAND_WEIGHT_MAX * generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, unit_funcs, ff_deg=ff_deg, tempering=[1.] * len(ff_deg))

### RUN_TEST function

def run_test(m, output_dir_name, show_connectivity=True, repeats=1, n_show_only=None,
    add_noise=True, dropouts=[{'E': 0, 'I': 0}], w_r_e=None, w_r_i=None, epochs=500):

    output_dir = f'./figures/{output_dir_name}'
    os.makedirs(output_dir)

    robustness_output_dir = f'./robustness/{output_dir_name}'
    os.makedirs(robustness_output_dir)
    
    w_u_e = np.diag(np.ones(m.N_DRIVING_CELLS)) * m.W_U_E
    
    ## input weights
    w_u = {
        # localized inputs to trigger activation from start of chain
        'E': np.block([
            [ w_u_e, np.zeros([m.N_DRIVING_CELLS, m.N_EXC + m.N_SILENT + m.N_INH]) ],
            [ np.zeros([m.N_EXC + m.N_SILENT + m.N_INH - m.N_DRIVING_CELLS, m.N_EXC + m.N_SILENT + m.N_INH + m.N_DRIVING_CELLS]) ],
        ]),

        'I': np.zeros((m.N_EXC + m.N_SILENT + m.N_INH , m.N_DRIVING_CELLS + m.N_EXC + m.N_SILENT + m.N_INH)),

        'A': np.zeros((m.N_EXC + m.N_SILENT + m.N_INH, m.N_DRIVING_CELLS + m.N_EXC + m.N_SILENT + m.N_INH)),
    }

    connectivity = np.ones((m.N_EXC, m.N_EXC))

    def solid_unit_func():
        return np.ones((m.PROJECTION_NUM, m.PROJECTION_NUM))

    def rand_unit_func():
        return np.random.rand(m.PROJECTION_NUM, m.PROJECTION_NUM)

    if w_r_e is None:
        w_e_e_r = generate_exc_ff_chain(m)

        w_e_e_r = np.where(np.random.rand(m.N_EXC + m.N_SILENT, 1) < 0.5, np.random.rand(m.N_EXC + m.N_SILENT, 1) * 0.3, 1) * w_e_e_r

        np.fill_diagonal(w_e_e_r, 0.)

        con_per_i = m.E_I_CON_PER_LINK * m.N_EXC / m.PROJECTION_NUM
        e_i_r = rand_per_row_mat(int(con_per_i), (m.N_INH, m.N_EXC))
        s_e_r = rand_per_row_mat(int(0.1 * m.N_SILENT), (m.N_EXC, m.N_SILENT))

        # e_i_r += np.where(np.random.rand(m.N_EXC, m.N_INH) > (1. - 0.05 * 150. / m.N_INH), np.random.rand(m.N_EXC, m.N_INH), 0.)

        w_r_e = np.block([
            [ w_e_e_r, s_e_r * m.W_INITIAL / m.PROJECTION_NUM, np.zeros((m.N_EXC, m.N_INH)) ],
            [ np.zeros((m.N_SILENT, m.N_EXC + m.N_SILENT + m.N_INH)) ],
            [ e_i_r * m.W_E_I_R,  np.zeros((m.N_INH, m.N_INH + m.N_SILENT)) ],
        ])

    if w_r_i is None:

        i_e_r = mat_1_if_under_val(m.I_E_CON_PROB, (m.N_EXC, m.N_INH))

        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC + m.N_SILENT)), i_e_r * m.W_I_E_R ],
            [ np.zeros((m.N_SILENT + m.N_INH, m.N_EXC + m.N_SILENT + m.N_INH)) ],
        ])
    
    ## recurrent weights
    w_r = {
        'E': w_r_e,
        'I': w_r_i,
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_SILENT + m.N_INH)) ],
            [ np.zeros((m.N_SILENT + m.N_INH, m.N_EXC + m.N_SILENT + m.N_INH)) ],
        ]),
    }
    
    all_rsps = []

    # run simulation for same set of parameters
    for rp_idx in range(repeats):
        show_trial = (type(n_show_only) is int and rp_idx < n_show_only)

        rsps_for_trial = []
        
        for d_idx, dropout in enumerate(dropouts):

            e_cell_fr_setpoints = None
            e_cell_pop_fr_setpoint = None
            active_cells_pre_dropout_mask = None
            surviving_cell_indices = None

            w_r_copy = copy(w_r)

            for i_e in range(epochs):

                # if i_e == m.DROPOUT_ITER:
                #     w_r_copy['E'][:, :(m.N_EXC + m.N_SILENT)], surviving_cell_indices = dropout_on_mat(w_r_copy['E'][:, :(m.N_EXC + m.N_SILENT)], dropout['E'], min_idx=m.DROPOUT_MIN_IDX, max_idx=m.DROPOUT_MAX_IDX)

                t = np.arange(0, S.T, S.DT)

                ## external currents
                if add_noise:
                    i_ext = m.SGM_N/S.DT * np.random.randn(len(t), m.N_EXC + m.N_SILENT + m.N_INH) + m.I_EXT_B
                else:
                    i_ext = m.I_EXT_B * np.ones((len(t), m.N_EXC + m.N_SILENT + m.N_INH))

                ## inp spks
                spks_u_base = np.zeros((len(t), m.N_DRIVING_CELLS + m.N_EXC + m.N_SILENT + m.N_INH), dtype=int)

                # trigger inputs
                activation_times = np.zeros((len(t), m.N_DRIVING_CELLS))
                for t_ctr in np.arange(0, S.T, 1./m.DRIVING_HZ):
                    activation_times[int(t_ctr/S.DT), :] = 1

                np.concatenate([np.random.poisson(m.DRIVING_HZ * S.DT, size=(len(t), 1)) for i in range(m.N_DRIVING_CELLS)], axis=1)
                spks_u = copy(spks_u_base)
                spks_u[:, :m.N_DRIVING_CELLS] = np.zeros((len(t), m.N_DRIVING_CELLS))
                burst_t = np.arange(0, 5 * int(m.BURST_T / S.DT), int(m.BURST_T / S.DT))

                for t_idx, driving_cell_idx in zip(*activation_times.nonzero()):
                    input_noise_t = np.array(np.random.normal(scale=m.INPUT_STD / S.DT), dtype=int)
                    try:
                        spks_u[burst_t + t_idx + input_noise_t, driving_cell_idx] = 1
                    except IndexError as e:
                        pass

                def create_prop(prop_exc, prop_inh):
                    return cc([prop_exc * np.ones(m.N_EXC), prop_inh * np.ones(m.N_INH)])

                c_m = create_prop(m.C_M_E, m.C_M_I)
                g_l = create_prop(m.G_L_E, m.G_L_I)
                e_l = create_prop(m.E_L_E, m.E_L_I)
                v_th = create_prop(m.V_TH_E, m.V_TH_I)
                e_r = create_prop(m.E_R_E, m.E_R_I)
                t_r = create_prop(m.T_R_E, m.T_R_I)
                i_b = create_prop(1, 0).astype(bool)

                ntwk = LIFNtwkG(
                    c_m=c_m,
                    g_l=g_l,
                    e_l=e_l,
                    v_th=v_th,
                    v_r=e_r,
                    t_r=t_r,
                    e_s={'E': M.E_E, 'I': M.E_I, 'A': M.E_A},
                    t_s={'E': M.T_E, 'I': M.T_E, 'A': M.T_A},
                    w_r=w_r_copy,
                    w_u=w_u,
                    i_b=i_b,
                    f_b=M.F_B,
                    t_b=M.T_B,
                    sigma_b=M.SIGMA_B,
                )

                clamp = Generic(v={0: np.repeat(m.E_L_E, m.N_EXC + m.N_SILENT + m.N_INH)}, spk={})

                # run smln
                rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext, spks_u=spks_u)

                scale = 0.8
                gs = gridspec.GridSpec(3, 1)
                fig = plt.figure(figsize=(9 * scale, 9 * scale), tight_layout=True)
                axs = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[2])]

                spks_for_e_cells = rsp.spks[:, :(m.N_EXC + m.N_SILENT)]
                if surviving_cell_indices is not None:
                    spks_for_e_cells[:, ~(surviving_cell_indices.astype(bool))] = 0

                raster = np.stack([rsp.spks_t, rsp.spks_c])
                inh_raster = raster[:, raster[1, :] > (m.N_EXC + m.N_SILENT)]

                if active_cells_pre_dropout_mask is not None:
                    exc_cells_initially_active = copy(spks_for_e_cells)
                    exc_cells_initially_active[:, ~active_cells_pre_dropout_mask] = 0
                    exc_cells_initially_active = np.stack(exc_cells_initially_active.nonzero())

                    exc_cells_newly_active = copy(spks_for_e_cells)
                    exc_cells_newly_active[:, active_cells_pre_dropout_mask] = 0
                    exc_cells_newly_active = np.stack(exc_cells_newly_active.nonzero())

                    axs[0].scatter(exc_cells_initially_active[0, :] * S.DT * 1000, exc_cells_initially_active[1, :], s=1, c='black', zorder=0, alpha=0.2)
                    axs[0].scatter(exc_cells_newly_active[0, :] * S.DT * 1000, exc_cells_newly_active[1, :], s=1, c='green', zorder=1, alpha=1)
                else:
                    exc_raster = raster[:, raster[1, :] < (m.N_EXC + m.N_SILENT)]

                    axs[0].scatter(exc_raster[0, :] * 1000, exc_raster[1, :], s=1, c='black', zorder=0, alpha=1)

                axs[0].scatter(inh_raster[0, :] * 1000, inh_raster[1, :], s=1, c='red', zorder=0, alpha=1)

                axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
                axs[0].set_xlim(0, 0.4 * 1000)
                axs[0].set_ylabel('Cell Index')
                axs[0].set_xlabel('Time (ms)')

                for i in range(len(axs)):
                    set_font_size(axs[i], 14)
                fig.savefig(f'{output_dir}/{d_idx}_{zero_pad(i_e, 4)}.png')

                first_spk_times = process_single_activation(exc_raster, m)

                sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{d_idx}_eidx_{zero_pad(i_e, 4)}', {
                    'first_spk_times': first_spk_times,
                    'exc_raster': exc_raster,
                    'inh_raster': inh_raster,
                })


def quick_plot(m, run_title='', w_r_e=None, w_r_i=None, repeats=1, show_connectivity=True, n_show_only=None, add_noise=True, dropouts=[{'E': 0, 'I': 0}]):
    output_dir_name = f'{run_title}_{time_stamp(s=True)}:{zero_pad(int(np.random.rand() * 9999), 4)}'

    run_test(m, output_dir_name=output_dir_name, show_connectivity=show_connectivity,
                        repeats=repeats, n_show_only=n_show_only, add_noise=add_noise, dropouts=dropouts,
                        w_r_e=w_r_e, w_r_i=w_r_i, epochs=S.EPOCHS)

def process_single_activation(exc_raster, m):
    # extract first spikes
    first_spk_times = np.nan * np.ones(m.N_EXC)
    for i in range(exc_raster.shape[1]):
        nrn_idx = int(exc_raster[1, i])
        if np.isnan(first_spk_times[nrn_idx]):
            first_spk_times[nrn_idx] = exc_raster[0, i]
    return first_spk_times

def load_weight_matrices(direc, num):
    file_names = sorted(all_files_from_dir(direc))
    file = file_names[num]
    loaded = sio.loadmat(os.path.join(direc, file))
    return loaded['w_r_e'], loaded['w_r_i']

def clip(f, n=1):
    f_str = str(f)
    f_str = f_str[:(f_str.find('.') + 1 + n)]
    return f_str

title = f'{args.title[0]}_ff_{clip(M.W_INITIAL / (0.26 * 0.004))}_pf_{clip(M.CON_PROB_FF, 2)}_pr_{clip(M.CON_PROB_R, 2)}_eir_{clip(M.W_E_I_R * 1e5)}_ier_{clip(M.W_I_E_R * 1e5)}'

for i in range(1):
    all_rsps = quick_plot(M, run_title=title, dropouts=[
        {'E': 0, 'I': 0},
    ])