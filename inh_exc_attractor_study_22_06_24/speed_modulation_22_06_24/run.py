from copy import deepcopy as copy
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import stats
import pandas as pd
import pickle
from collections import OrderedDict
import os
from scipy.ndimage.interpolation import shift
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.sparse import csc_matrix, csr_matrix, kron
from functools import reduce, partial
import argparse
import time
import tracemalloc

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

matplotlib.use('agg')

cmap = cm.viridis
cmap.set_under(color='white')

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--rng_seed', metavar='r', type=int, nargs=1)
parser.add_argument('--dropout_per', metavar='d', type=float, nargs=1)
parser.add_argument('--dropout_iter', metavar='di', type=int, nargs=1)

parser.add_argument('--w_ee', metavar='ee', type=float, nargs=1)
parser.add_argument('--beta', metavar='b', type=float, nargs=1)
parser.add_argument('--w_ei', metavar='ei', type=float, nargs=1)
parser.add_argument('--w_ie', metavar='ie', type=float, nargs=1)
parser.add_argument('--w_u', metavar='u', type=float, nargs=1)

parser.add_argument('--index', metavar='I', type=int, nargs=1)

parser.add_argument('--load_run', metavar='L', type=str, nargs=2)


args = parser.parse_args()

print(args)

# PARAMS
## NEURON AND NETWORK MODEL
M = Generic(
    # Excitatory membrane
    C_M_E=1e-6,  # membrane capacitance
    G_L_E=0.25e-3,  # membrane leak conductance (T_M (s) = C_M (F/cm^2) / G_L (S/cm^2))
    E_L_E=-.07,  # membrane leak potential (V)
    V_TH_E=-.043,  # membrane spike threshold (V)
    T_R_E=1e-3,  # refractory period (s)
    E_R_E=-0.065, # reset voltage (V)
    
    # Inhibitory membrane
    C_M_I=1e-6,
    G_L_I=.5e-3, 
    E_L_I=-.057,
    V_TH_I=-.043,
    T_R_I=1e-3, #0.25e-3,
    E_R_I=-.055, # reset voltage (V)
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.09, E_A=-.07, T_E=.004, T_I=.004, T_A=.006,
    
    N_EXC=200,
    N_UVA=0,
    N_INH=200,
    M=20,
    
    # Input params
    DRIVING_HZ=1, # 2 Hz lambda Poisson input to system
    N_DRIVING_CELLS=10,
    PROJECTION_NUM=10,
    INPUT_STD=1e-3,
    BURST_T=1.5e-3,
    INPUT_DELAY=10e-3,
    
    # OTHER INPUTS
    SGM_N=10e-11,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input

    # Connection probabilities
    CON_PROB_R=0.,
    E_I_CON_PROB=0.05,
    I_E_CON_PROB=1.,

    # Weights
    W_E_I_R=args.w_ei[0],
    W_I_E_R=args.w_ie[0],
    W_A=0,
    W_E_E_F=args.w_ee[0],
    BETA=args.beta[0],
    W_U=args.w_u[0],
    W_E_E_R_MIN=1e-8,
    W_E_E_R_MAX=10e-4,
    SUPER_SYNAPSE_SIZE=1.5e-3,

    # Dropout params
    DROPOUT_MIN_IDX=0,
    DROPOUT_MAX_IDX=0, # set elsewhere
    DROPOUT_ITER=args.dropout_iter[0],
    DROPOUT_SEV=args.dropout_per[0],
)

S = Generic(RNG_SEED=args.rng_seed[0], DT=0.05e-3, T=500e-3, EPOCHS=20)
np.random.seed(S.RNG_SEED)

M.SUMMED_W_E_E_R_MAX = M.W_E_E_F

M.DROPOUT_MAX_IDX = M.N_EXC


# a = sample_sphere(2000)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(a[:, 0], a[:, 1], a[:, 2])

# fig.savefig('sphere.png')

## SMLN

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)

def ff_unit_func(m, p):
    connectivity = gaussian_if_under_val(p, (m.PROJECTION_NUM, m.PROJECTION_NUM), 1, 0)
    return connectivity

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

def gen_continuous_network(size, m):
    w = m.W_E_E_R / m.PROJECTION_NUM

    active_cell_mask = np.random.rand(size) > args.silent_fraction[0]
    cont_idx_steps = np.random.rand(size)
    cont_idx = np.array([np.sum(cont_idx_steps[:i]) for i in range(cont_idx_steps.shape[0])]) * (6 / size)

    active_inactive_pairings = np.outer(active_cell_mask, active_cell_mask).astype(bool)
    cont_idx_dists = cont_idx.reshape(cont_idx.shape[0], 1) - cont_idx.reshape(1, cont_idx.shape[0])

    def exp_if_pos(dist, tau):
        return np.where(np.logical_and(dist >= 0, dist < 0.4), 1., 0) # np.exp(-dist/tau), 0)

    sequence_weights = np.where(active_inactive_pairings, (0.5 + 0.4 * args.silent_fraction[0]) * w * exp_if_pos(cont_idx_dists, 0.15), exp_if_under_val(0.05, (size, size), 0.1 * w))
    sequence_delays = np.abs(cont_idx_dists)
    np.fill_diagonal(sequence_delays, 0)

    weights = np.zeros((m.N_EXC, m.N_EXC))
    weights[:size, :size] = sequence_weights

    delays = np.zeros((m.N_EXC, m.N_EXC))
    delays[:size, :size] = sequence_delays

    all_active_inactive_pairings = np.zeros((m.N_EXC, m.N_EXC)).astype(bool)
    all_active_inactive_pairings[:size, :size] = active_inactive_pairings

    undefined_delays = np.logical_or(weights < m.W_E_E_R_MIN, ~all_active_inactive_pairings)

    delays[undefined_delays] = 1/3 * np.random.rand(np.count_nonzero(undefined_delays))

    delays = delays / np.mean(delays[weights > m.W_E_E_R_MIN])

    return weights, delays

### RUN_TEST function
def run_test(m, output_dir_name, n_show_only=None, add_noise=True, dropout={'E': 0, 'I': 0},
    w_r_e=None, w_r_i=None, epochs=500, e_cell_pop_fr_setpoint=None):

    output_dir = f'./figures/{output_dir_name}'
    os.makedirs(output_dir)

    robustness_output_dir = f'./robustness/{output_dir_name}'
    os.makedirs(robustness_output_dir)

    sampled_cell_output_dir = f'./sampled_cell_rasters/{output_dir_name}'
    os.makedirs(sampled_cell_output_dir)
    
    w_u_proj = np.diag(np.ones(m.N_DRIVING_CELLS)) * 1e-4
    w_u_uva = np.diag(np.ones(m.N_EXC - m.N_DRIVING_CELLS)) * m.W_U

    w_u_e = np.zeros([m.N_EXC, m.N_EXC])
    w_u_e[:m.N_DRIVING_CELLS, :m.N_DRIVING_CELLS] += w_u_proj
    w_u_e[m.N_DRIVING_CELLS:m.N_EXC, m.N_DRIVING_CELLS:m.N_EXC] += w_u_uva

    ## input weights
    w_u = {
        # localized inputs to trigger activation from start of chain
        'E': np.block([
            [ w_u_e ],
            [ np.zeros([m.N_INH, m.N_EXC]) ],
        ]),

        'I': np.zeros((m.N_EXC + m.N_INH, m.N_EXC)),

        'A': np.zeros((m.N_EXC + m.N_INH, m.N_EXC)),
    }

    if w_r_e is None:
        temperings = [m.BETA, 1]
        ff_deg = [0, 1]
        unit_funcs = []

        for l_idx in range(len(ff_deg)):
            unit_funcs.append(partial(ff_unit_func, m=m, p=1.))

        w_e_e_r = generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, unit_funcs, ff_deg, temperings)
        np.fill_diagonal(w_e_e_r, 0.)
        w_e_e_r = m.W_E_E_F / m.PROJECTION_NUM * w_e_e_r

        temperings = [1]
        ff_deg = [0]
        unit_funcs = []

        for l_idx in range(len(ff_deg)):
            unit_funcs.append(partial(ff_unit_func, m=m, p=1.))

        w_e_i_r = generate_ff_chain(m.N_INH, m.PROJECTION_NUM, unit_funcs, ff_deg, temperings)
        np.fill_diagonal(w_e_i_r, 0.)
        w_e_i_r = m.W_E_I_R / m.PROJECTION_NUM * w_e_i_r

        w_r_e = np.block([
            [ w_e_e_r, np.zeros((m.N_EXC, m.N_INH)) ],
            [ w_e_i_r,  np.zeros((m.N_INH, m.N_INH)) ],
        ])

    if w_r_i is None:
        temperings = [1, 1]
        ff_deg = [0, 1]
        unit_funcs = []

        for l_idx in range(len(ff_deg)):
            unit_funcs.append(partial(ff_unit_func, m=m, p=1.))

        w_i_e_r = np.ones((m.N_EXC, m.N_INH)) - generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, unit_funcs, ff_deg, temperings)
        np.fill_diagonal(w_i_e_r, 0.)
        w_i_e_r = m.W_I_E_R / m.PROJECTION_NUM * w_i_e_r 


        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC + m.N_UVA)), w_i_e_r],
            [ np.zeros((m.N_UVA + m.N_INH, m.N_EXC + m.N_UVA + m.N_INH)) ],
        ])
    
    ## recurrent weights
    w_r = {
        'E': w_r_e,
        'I': w_r_i,
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_UVA + m.N_INH)) ],
            [ np.zeros((m.N_UVA + m.N_INH, m.N_EXC + m.N_UVA + m.N_INH)) ],
        ]),
    }

    ee_connectivity = np.where(w_r_e[:(m.N_EXC), :(m.N_EXC + m.N_UVA)] > 0, 1, 0)

    def int_delay(d):
        return np.max([int(d / S.DT), 1])

    pairwise_spk_delays = np.block([
        [int_delay(0) * np.ones((m.N_EXC, m.N_EXC)), np.ones((m.N_EXC, m.N_UVA)), int_delay(0) * np.ones((m.N_EXC, m.N_INH))],
        [int_delay(0) * np.ones((m.N_INH + m.N_UVA, m.N_EXC + m.N_INH + m.N_UVA))],
    ]).astype(int)

    # turn pairwise delays into list of cells one cell is synapsed to with some delay tau
   
    def make_delay_map(w_r):
        delay_map = {}
        summed_w_r_abs = np.sum(np.stack([np.abs(w_r[syn]) for syn in w_r.keys()]), axis=0)
        for i in range(pairwise_spk_delays.shape[1]):
            cons = summed_w_r_abs[:, i].nonzero()[0]
            delay_map[i] = (pairwise_spk_delays[cons, i], cons)
        return delay_map

    delay_map = make_delay_map(w_r)


    def create_prop(prop_exc, prop_inh):
        return cc([prop_exc * np.ones(m.N_EXC + m.N_UVA), prop_inh * np.ones(m.N_INH)])

    c_m = create_prop(m.C_M_E, m.C_M_I)
    g_l = create_prop(m.G_L_E, m.G_L_I)
    e_l = create_prop(m.E_L_E, m.E_L_I)
    v_th = create_prop(m.V_TH_E, m.V_TH_I)
    e_r = create_prop(m.E_R_E, m.E_R_I)
    t_r = create_prop(m.T_R_E, m.T_R_I)


    sampled_e_cell_rasters = []
    e_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_EXC).astype(int))
    sampled_i_cell_rasters = []
    i_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_INH + m.N_EXC).astype(int))

    w_r_copy = copy(w_r)

    # tracemalloc.start()

    # snapshot = None
    # last_snapshot = tracemalloc.take_snapshot()

    surviving_cell_mask = None

    for i_e in range(epochs):

        progress = f'{i_e / epochs * 100}'
        progress = progress[: progress.find('.') + 2]
        print(f'{progress}% finished')

        start = time.time()

        if i_e == m.DROPOUT_ITER:
            w_r_copy['E'][:(m.N_EXC + m.N_UVA + m.N_INH), :m.N_EXC], surviving_cell_mask = dropout_on_mat(w_r_copy['E'][:(m.N_EXC + m.N_UVA + m.N_INH), :m.N_EXC], dropout['E'])
            surviving_cell_mask = surviving_cell_mask.astype(bool)
            ee_connectivity = np.where(w_r_copy['E'][:(m.N_EXC), :(m.N_EXC + m.N_UVA)] > 0, 1, 0)

        t = np.arange(0, S.T, S.DT)

        ## external currents
        if add_noise:
            i_ext = m.SGM_N/S.DT * np.random.randn(len(t), m.N_EXC + m.N_UVA + m.N_INH) + m.I_EXT_B
        else:
            i_ext = m.I_EXT_B * np.ones((len(t), m.N_EXC + m.N_UVA + m.N_INH))

        ## inp spks
        spks_u_base = np.zeros((len(t), m.N_EXC), dtype=int)

        # trigger inputs
        activation_times = np.zeros((len(t), m.N_DRIVING_CELLS))
        for t_ctr in np.arange(0, S.T, 1./m.DRIVING_HZ):
            activation_times[int(t_ctr/S.DT), :] = 1

        spks_u = copy(spks_u_base)
        spks_u[:, :m.N_DRIVING_CELLS] = np.zeros((len(t), m.N_DRIVING_CELLS))
        burst_t = np.arange(0, 5 * int(m.BURST_T / S.DT), int(m.BURST_T / S.DT))

        trip_spk_hist = [[] for n_e in range(m.N_EXC)]

        for t_idx, driving_cell_idx in zip(*activation_times.nonzero()):
            input_noise_t = np.array(np.random.normal(scale=m.INPUT_STD / S.DT), dtype=int)
            try:
                spks_u[burst_t + t_idx + input_noise_t + int(m.INPUT_DELAY / S.DT), driving_cell_idx] = 1
            except IndexError as e:
                pass

        def make_poisson_input(dur=0.5, offset=0.05):
            x = np.zeros(len(t))
            if dur + offset > S.T:
                dur = S.T - offset

            x[int(offset/S.DT):int(offset/S.DT) + int(dur/S.DT)] = np.random.poisson(lam=800 * S.DT, size=int(dur/S.DT))
            return x

        spks_u[:, m.N_DRIVING_CELLS:m.N_EXC] = np.stack([make_poisson_input() for i in range(m.N_EXC - m.N_DRIVING_CELLS)]).T

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
            pairwise_spk_delays=pairwise_spk_delays,
            delay_map=delay_map,
        )

        clamp = Generic(v={0: e_l}, spk={})

        # run smln
        rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext, spks_u=spks_u)

        scale = 0.8
        gs = gridspec.GridSpec(14, 1)
        fig = plt.figure(figsize=(9 * scale, 30 * scale), tight_layout=True)
        axs = [
            fig.add_subplot(gs[:2]),
            fig.add_subplot(gs[2]),
            fig.add_subplot(gs[3]),
            fig.add_subplot(gs[4]),
            fig.add_subplot(gs[5]),
            fig.add_subplot(gs[6:8]),
            fig.add_subplot(gs[8:10]),
            fig.add_subplot(gs[10:12]),
            fig.add_subplot(gs[12:]),
        ]

        w_e_e_r_copy = w_r_copy['E'][:m.N_EXC, :m.N_EXC]
        if surviving_cell_mask is not None:
            w_e_e_r_copy = w_e_e_r_copy[surviving_cell_mask, :]

        # 0.05 * np.mean(w_e_e_r_copy.sum(axis=1)
        summed_w_bins, summed_w_counts = bin_occurrences(w_e_e_r_copy.sum(axis=1), bin_size=1e-4, max_val=0.004)
        axs[3].plot(summed_w_bins, summed_w_counts)
        axs[3].set_xlabel('Normalized summed synapatic weight')
        axs[3].set_ylabel('Counts')

        incoming_con_counts = np.count_nonzero(w_e_e_r_copy, axis=1)
        incoming_con_bins, incoming_con_freqs = bin_occurrences(incoming_con_counts, bin_size=1)
        axs[4].plot(incoming_con_bins, incoming_con_freqs)
        axs[4].set_xlabel('Number of incoming synapses per cell')
        axs[4].set_ylabel('Counts')

        min_ee_weight = w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)].min()
        graph_weight_matrix(w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)], 'w_e_e_r\n', ax=axs[5],
            v_min=min_ee_weight, v_max=m.W_E_E_R_MAX/5, cmap=cmap)
        graph_weight_matrix(w_r_copy['I'][:m.N_EXC, (m.N_EXC + m.N_UVA):], 'w_i_e_r\n', ax=axs[6], v_max=m.W_E_I_R, cmap=cmap)

        spks_for_e_cells = rsp.spks[:, :m.N_EXC]
        spks_for_i_cells = rsp.spks[:, (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH)]

        spks_received_for_e_cells = rsp.spks_received[:, :m.N_EXC, :m.N_EXC]
        spks_received_for_i_cells = rsp.spks_received[:, (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH), (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH)]

        spk_bins, freqs = bin_occurrences(spks_for_e_cells.sum(axis=0), max_val=800, bin_size=1)

        axs[1].bar(spk_bins, freqs, alpha=0.5)
        axs[1].set_xlabel('Spks per neuron')
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlim(-0.5, 30.5)
        # axs[1].set_ylim(0, m.N_EXC + m.N_SILENT)

        raster = np.stack([rsp.spks_t, rsp.spks_c])
        exc_raster = raster[:, raster[1, :] < m.N_EXC]
        inh_raster = raster[:, raster[1, :] >= (m.N_EXC + m.N_UVA)]

        spk_bins_i, freqs_i = bin_occurrences(spks_for_i_cells.sum(axis=0), max_val=800, bin_size=1)

        axs[2].bar(spk_bins_i, freqs_i, color='black', alpha=0.5, zorder=-1)
        axs[2].set_xlim(-0.5, 100)

        axs[0].scatter(exc_raster[0, :] * 1000, exc_raster[1, :], s=1, c='black', zorder=0, alpha=1)
        axs[0].scatter(inh_raster[0, :] * 1000, inh_raster[1, :] - m.N_UVA, s=1, c='red', zorder=0, alpha=1)

        axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
        axs[0].set_xlim(0, S.T * 1000)
        axs[0].set_ylabel('Cell Index')
        axs[0].set_xlabel('Time (ms)')

        for i in range(len(axs)):
            set_font_size(axs[i], 14)

        first_spk_times = process_single_activation(exc_raster, m)

        if i_e % 10 == 0:

            mean_initial_first_spk_time = np.nanmean(first_spk_times[50:60])
            mean_final_first_spk_time = np.nanmean(first_spk_times[160:170])
            prop_speed = (16 - 5) / (mean_final_first_spk_time - mean_initial_first_spk_time)
            spiking_idxs = np.nonzero(first_spk_times[50:170])[0]
            spks_for_spiking_idxs = spks_for_e_cells[:, spiking_idxs]

            print(prop_speed)

            temporal_widths = []

            for k in range(spks_for_spiking_idxs.shape[1]):
                temporal_widths.append(np.std(S.DT * np.nonzero(spks_for_spiking_idxs[:, k])[0]))
            avg_temporal_width = np.mean(temporal_widths)

            base_data_to_save = {
                'w_e_e': m.W_E_E_F,
                'beta': m.BETA,
                'w_e_i': m.W_E_I_R,
                'w_i_e': m.W_I_E_R,
                'w_u': m.W_U,
                'first_spk_times': first_spk_times,
                'spk_bins': spk_bins,
                'freqs': freqs,
                'exc_raster': exc_raster,
                'inh_raster': inh_raster,
                'prop_speed': prop_speed,
                'temporal_widths': temporal_widths,
                'avg_temporal_width': avg_temporal_width,
                # 'gs': rsp.gs,
            }

            if i_e >= m.DROPOUT_ITER:
                base_data_to_save['surviving_cell_mask'] = surviving_cell_mask

            if i_e % 10 == 0:
                update_obj = {
                    'w_r_e': rsp.ntwk.w_r['E'],
                    'w_r_i': rsp.ntwk.w_r['I'],
                }
                base_data_to_save.update(update_obj)

            sio.savemat(robustness_output_dir + '/' + f'title_{title}_idx_{zero_pad(i_e, 4)}', base_data_to_save)
        fig.savefig(f'{output_dir}/{zero_pad(i_e, 4)}.png')

        end = time.time()
        secs_per_cycle = f'{end - start}'
        secs_per_cycle = secs_per_cycle[:secs_per_cycle.find('.') + 2]
        print(f'{secs_per_cycle} s')

        plt.close('all')

        # snapshot = tracemalloc.take_snapshot()
        # if last_snapshot is not None:
        #     top_stats = snapshot.compare_to(last_snapshot, 'lineno')
        #     print("[ Top 3 differences ]")
        #     for stat in top_stats[:3]:
        #         print(stat)



def quick_plot(m, run_title='', w_r_e=None, w_r_i=None, n_show_only=None, add_noise=True, dropout={'E': 0, 'I': 0}, e_cell_pop_fr_setpoint=None):
    output_dir_name = f'{run_title}_{time_stamp(s=True)}:{zero_pad(int(np.random.rand() * 9999), 4)}'

    run_test(m, output_dir_name=output_dir_name, n_show_only=n_show_only, add_noise=add_noise, dropout=dropout,
                        w_r_e=w_r_e, w_r_i=w_r_i, epochs=S.EPOCHS, e_cell_pop_fr_setpoint=e_cell_pop_fr_setpoint)

def process_single_activation(exc_raster, m):
    # extract first spikes
    first_spk_times = np.nan * np.ones(m.N_EXC + m.N_UVA)
    for i in range(exc_raster.shape[1]):
        nrn_idx = int(exc_raster[1, i])
        if np.isnan(first_spk_times[nrn_idx]):
            first_spk_times[nrn_idx] = exc_raster[0, i]
    return first_spk_times

def load_previous_run(direc, num):
    file_names = sorted(all_files_from_dir(direc))
    file = file_names[num]
    loaded = sio.loadmat(os.path.join(direc, file))
    return loaded

def clip(f, n=1):
    f_str = str(f)
    f_str = f_str[:(f_str.find('.') + 1 + n)]
    return f_str

title = f'{args.title[0]}'

w_r_e = None
w_r_i = None

if args.load_run is not None:
    loaded = load_previous_run(os.path.join('./robustness', args.load_run[0]), int(args.load_run[1]))
    w_r_e = np.array(loaded['w_r_e'].todense())
    w_r_i = np.array(loaded['w_r_i'].todense())

quick_plot(M, run_title=title, dropout={'E': M.DROPOUT_SEV, 'I': 0}, w_r_e=w_r_e, w_r_i=w_r_i)


