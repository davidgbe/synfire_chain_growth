from copy import deepcopy as copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--alpha_1', metavar='a1', type=float, nargs=1)
parser.add_argument('--alpha_2', metavar='a2', type=float, nargs=1)
parser.add_argument('--beta', metavar='b', type=float, nargs=1)
parser.add_argument('--gamma', metavar='c', type=float, nargs=1)
parser.add_argument('--fr_single_line_attr', metavar='s', type=int, nargs=1)
parser.add_argument('--rng_seed', metavar='r', type=int, nargs=1)
parser.add_argument('--load_mat', metavar='l', type=str, nargs=1)
parser.add_argument('--dropout_per', metavar='d', type=float, nargs=1)


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
    T_R_E=2e-3,  # refractory period (s)
    E_R_E=-0.067, # reset voltage (V)
    
    # Inhibitory membrane
    C_M_I=1e-6,
    G_L_I=.4e-3, 
    E_L_I=-.057,
    V_TH_I=-.043,
    T_R_I=0.25e-3,
    E_R_I=-.057, # reset voltage (V)
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.09, E_A=-.07, T_E=.004, T_I=.004, T_A=.006,
    
    N_EXC=300,
    N_SILENT=0,
    N_INH=200,
    M=20,
    
    # Input params
    DRIVING_HZ=2, # 2 Hz lambda Poisson input to system
    N_DRIVING_CELLS=20,
    PROJECTION_NUM=20,
    INPUT_STD=1e-3,
    BURST_T=1.5e-3,
    INPUT_DELAY=50e-3,
    
    # OTHER INPUTS
    SGM_N=10e-10,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input

    # Connection probabilities
    MEAN_N_CONS_PER_CELL=45,
    SYN_PROP_DIST_EXP=1.7,
    CON_PROB_FF_CONST=1,
    CON_PROB_R=0.,
    E_I_CON_PROB=0.25,
    I_E_CON_PROB=0.6,

    # Weights
    W_E_I_R=2.5e-5,
    W_E_I_R_MAX=10e-5,
    W_I_E_R=0.9e-5,
    W_A=0,
    W_E_E_R=0.26 * 0.004 * 1.3,
    W_E_E_R_MAX=0.26 * 0.004 * 1.3,
    CELL_OUTPUT_MAX=0.26 * 0.004 * 1.3 * 2,

    # Dropout params
    DROPOUT_MIN_IDX=0,
    DROPOUT_ITER=10000,
    DROPOUT_SEV=0,

    # Synaptic plasticity params
    TAU_STDP_PAIR=30e-3,
    SINGLE_CELL_FR_SETPOINT_MIN=6,
    SINGLE_CELL_FR_SETPOINT_MIN_STD=2,
    SINGLE_CELL_LINE_ATTR=bool(args.fr_single_line_attr[0]),
    ETA=0.3,
    ALPHA_1=args.alpha_1[0], #3e-2
    ALPHA_2=args.alpha_2[0],
    BETA=args.beta[0], #1e-3,
    GAMMA=args.gamma[0], #1e-4,
)

S = Generic(RNG_SEED=args.rng_seed[0], DT=0.22e-3, T=150e-3, EPOCHS=1000)
np.random.seed(S.RNG_SEED)

M.CON_PROBS_FF = np.exp(-1 * np.arange(M.N_EXC / M.PROJECTION_NUM) / M.CON_PROB_FF_CONST)

M.W_U_E = M.W_E_E_R / M.PROJECTION_NUM * 1.55

M.CUT_IDX_TAU_PAIR = int(2 * M.TAU_STDP_PAIR / S.DT)
M.KERNEL_PAIR = np.exp(-np.arange(M.CUT_IDX_TAU_PAIR) * S.DT / M.TAU_STDP_PAIR).astype(float)

M.DROPOUT_MAX_IDX = M.N_EXC + M.N_SILENT

## SMLN

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)


### RUN_TEST function

def run_test(m, output_dir_name, show_connectivity=True, repeats=1, n_show_only=None,
    add_noise=True, dropouts=[{'E': 0, 'I': 0}], w_r_e=None, w_r_i=None, epochs=500):

    output_dir = f'./figures/{output_dir_name}'
    os.makedirs(output_dir)

    robustness_output_dir = f'./robustness/{output_dir_name}'
    os.makedirs(robustness_output_dir)

    sampled_cell_output_dir = f'./sampled_cell_rasters/{output_dir_name}'
    os.makedirs(sampled_cell_output_dir)
    
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


    if w_r_e is None:
        w_e_e_r = np.random.rand(m.N_EXC, m.N_EXC)
        w_e_e_r[w_e_e_r > 0.3] = 0
        w_e_e_r *= (m.W_E_E_R * 0.03 / 0.3)
        np.fill_diagonal(w_e_e_r, 0.)

        connectivity = np.where(w_e_e_r > 0, 1, 0)

        e_i_r = gaussian_if_under_val(m.E_I_CON_PROB, (m.N_INH, m.N_EXC), m.W_E_I_R, 0.3 * m.W_E_I_R)

        s_e_r = rand_per_row_mat(0, (m.N_EXC, m.N_SILENT))

        w_r_e = np.block([
            [ w_e_e_r, s_e_r * m.W_E_E_R / m.PROJECTION_NUM, np.zeros((m.N_EXC, m.N_INH)) ],
            [ np.zeros((m.N_SILENT, m.N_EXC + m.N_SILENT + m.N_INH)) ],
            [ e_i_r,  np.zeros((m.N_INH, m.N_INH + m.N_SILENT)) ],
        ])

    if w_r_i is None:

        i_e_r = gaussian_if_under_val(m.I_E_CON_PROB, (m.N_EXC, m.N_INH), 0.5 * m.W_I_E_R, 0.3 * m.W_I_E_R)

        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC + m.N_SILENT)), i_e_r],
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

    def create_prop(prop_exc, prop_inh):
        return cc([prop_exc * np.ones(m.N_EXC), prop_inh * np.ones(m.N_INH)])

    c_m = create_prop(m.C_M_E, m.C_M_I)
    g_l = create_prop(m.G_L_E, m.G_L_I)
    e_l = create_prop(m.E_L_E, m.E_L_I)
    v_th = create_prop(m.V_TH_E, m.V_TH_I)
    e_r = create_prop(m.E_R_E, m.E_R_I)
    t_r = create_prop(m.T_R_E, m.T_R_I)
    
    all_rsps = []

    # run simulation for same set of parameters
    for rp_idx in range(repeats):
        show_trial = (type(n_show_only) is int and rp_idx < n_show_only)

        rsps_for_trial = []
        
        for d_idx, dropout in enumerate(dropouts):

            e_cell_fr_setpoints = np.ones(m.N_EXC)
            e_cell_pop_fr_setpoint = 2 * m.PROJECTION_NUM

            sampled_e_cell_rasters = []
            e_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_EXC).astype(int))
            sampled_i_cell_rasters = []
            i_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_INH + m.N_EXC).astype(int))

            w_r_copy = copy(w_r)

            for i_e in range(epochs):

                progress = f'{i_e / epochs * 100}'
                progress = progress[: progress.find('.') + 2]
                print(f'{progress}% finished')

                start = time.time()

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
                        spks_u[burst_t + t_idx + input_noise_t + int(m.INPUT_DELAY / S.DT), driving_cell_idx] = 1
                    except IndexError as e:
                        pass

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
                    plasticity_indices=np.arange(m.N_EXC),
                    connectivity=connectivity,
                    W_max=m.W_E_E_R_MAX,
                    m=m.M,
                    output=False,
                    output_freq=100000,
                    weight_update=False,
                )

                clamp = Generic(v={0: e_l}, spk={})

                # run smln
                rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext,
                                output_dir_name=f'{output_dir_name}_{rp_idx}_{d_idx}', spks_u=spks_u,
                                dropouts=[], m=m, repairs=[])


                # fig = plt.figure(figsize=(4, 4), tight_layout=True)
                # ax = fig.add_subplot()
                # inh_voltages = rsp.vs[int((0.04 + m.INPUT_DELAY)/S.DT):int((0.055 + m.INPUT_DELAY)/S.DT), m.N_EXC:]
                # mean_inh_voltage = inh_voltages.mean()

                # for i in range(inh_voltages.shape[0]):
                #     b, f = bin_occurrences(inh_voltages[i, :], bin_size=0.05 * np.abs(mean_inh_voltage), min_val=m.E_I, max_val=-0.04)
                #     ax.plot(b, f)

                # b, f = bin_occurrences(rsp.vs[int(50e-3 / S.DT), m.N_EXC:], bin_size=0.05 * np.abs(mean_inh_voltage), min_val=m.E_I, max_val=-0.04)
                # ax.plot(b, f/2, color='black', lw=2)
                # fig.savefig('inh_volt_dists.png')

                sampled_e_cell_rasters.append(rsp.spks[int((m.INPUT_DELAY + 20e-3)/S.DT):, e_cell_sample_idxs])
                sampled_i_cell_rasters.append(rsp.spks[int((m.INPUT_DELAY + 20e-3)/S.DT):, i_cell_sample_idxs])

                sampled_trial_number = 10
                if i_e % sampled_trial_number == 0 and i_e != 0:
                    fig = plt.figure(figsize=(8, 8), tight_layout=True)
                    ax = fig.add_subplot()
                    base_idx = 0
                    for rasters_for_cell_type in [sampled_e_cell_rasters, sampled_i_cell_rasters]:
                        for rendition_num in range(len(rasters_for_cell_type)):
                            for cell_idx in range(rasters_for_cell_type[rendition_num].shape[1]):
                                spk_times_for_cell = np.nonzero(rasters_for_cell_type[rendition_num][:, cell_idx])[0]
                                ax.scatter(spk_times_for_cell * S.DT * 1000, (base_idx + cell_idx * len(rasters_for_cell_type) + rendition_num) * np.ones(len(spk_times_for_cell)), s=3, marker='|')
                        base_idx += sampled_trial_number * rasters_for_cell_type[0].shape[1]
                    ax.set_xlim(0, 150)
                    ax.set_xlabel('Time (ms)')
                    sampled_e_cell_rasters = []
                    sampled_i_cell_rasters = []
                    fig.savefig(f'{sampled_cell_output_dir}/sampled_cell_rasters_{int(i_e / sampled_trial_number)}.png')

                scale = 0.8
                gs = gridspec.GridSpec(7, 1)
                fig = plt.figure(figsize=(9 * scale, 23 * scale), tight_layout=True)
                axs = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3]), fig.add_subplot(gs[4]), fig.add_subplot(gs[5:])]

                w_e_e_r_copy = w_r_copy['E'][:m.N_EXC, :m.N_EXC]

                summed_w_bins, summed_w_counts = bin_occurrences(w_e_e_r_copy.sum(axis=1), bin_size=0.05 * np.mean(w_e_e_r_copy.sum(axis=1)))
                axs[2].plot(summed_w_bins, summed_w_counts)
                axs[2].set_xlabel('Normalized summed synapatic weight')
                axs[2].set_ylabel('Counts')

                incoming_con_counts = np.count_nonzero(w_e_e_r_copy, axis=0)
                incoming_con_bins, incoming_con_freqs = bin_occurrences(incoming_con_counts, bin_size=1)
                axs[3].plot(incoming_con_bins, incoming_con_freqs)
                axs[3].set_xlabel('Number of incoming synapses per cell')
                axs[3].set_ylabel('Counts')

                graph_weight_matrix(w_e_e_r_copy, 'w_e_e_r\n', ax=axs[4], v_max=m.W_E_E_R * .25)

                spks_for_e_cells = rsp.spks[:, :(m.N_EXC + m.N_SILENT)]
                spks_for_i_cells = rsp.spks[:, (m.N_EXC + m.N_SILENT):(m.N_EXC + m.N_SILENT + m.N_INH)]

                spk_bins, freqs = bin_occurrences(spks_for_e_cells.sum(axis=0), max_val=800, bin_size=1)

                axs[1].bar(spk_bins, freqs, alpha=0.5)
                axs[1].set_xlabel('Spks per neuron')
                axs[1].set_ylabel('Frequency')
                axs[1].set_xlim(-0.5, 30.5)
                # axs[1].set_ylim(0, m.N_EXC + m.N_SILENT)

                raster = np.stack([rsp.spks_t, rsp.spks_c])
                inh_raster = raster[:, raster[1, :] > (m.N_EXC + m.N_SILENT)]

                spk_bins_i, freqs_i = bin_occurrences(spks_for_i_cells.sum(axis=0), max_val=800, bin_size=1)

                axs[1].bar(spk_bins_i, freqs_i, color='black', alpha=0.5)


                exc_raster = raster[:, raster[1, :] < (m.N_EXC + m.N_SILENT)]
                axs[0].scatter(exc_raster[0, :] * 1000, exc_raster[1, :], s=1, c='black', zorder=0, alpha=1)
                axs[0].scatter(inh_raster[0, :] * 1000, inh_raster[1, :], s=1, c='red', zorder=0, alpha=1)

                axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
                axs[0].set_xlim(m.INPUT_DELAY * 1000, 100)
                axs[0].set_ylabel('Cell Index')
                axs[0].set_xlabel('Time (ms)')

                for i in range(len(axs)):
                    set_font_size(axs[i], 14)
                fig.savefig(f'{output_dir}/{d_idx}_{zero_pad(i_e, 4)}.png')

                first_spk_times = process_single_activation(exc_raster, m)

                if i_e == 0:
                    sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{d_idx}_eidx_{zero_pad(i_e, 4)}', {
                        'first_spk_times': first_spk_times,
                        'w_r_e_summed': np.sum(rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC], axis=1),
                        'w_r_e_i_summed': np.sum(rsp.ntwk.w_r['E'][m.N_EXC:, :m.N_EXC], axis=1),
                        'spk_bins': spk_bins,
                        'freqs': freqs,
                        'exc_raster': exc_raster,
                        'inh_raster': inh_raster,
                    })
                else:
                    if i_e % 100 == 0:
                        e_cell_pop_fr_setpoint += m.PROJECTION_NUM

                    # filter e cell spks for start of bursts
                    def burst_kernel(spks):
                        if spks.shape[0] > 1 and np.count_nonzero(spks[:-1]) > 0:
                            return 0
                        else:
                            return spks[-1]

                    filtered_spks_for_e_cells = np.zeros(spks_for_e_cells.shape)
                    t_steps_in_burst = int(20e-3/S.DT)

                    for i_c in range(spks_for_e_cells.shape[1]):
                        for i_t in range(spks_for_e_cells.shape[0]):
                            idx_filter_start = (i_t - t_steps_in_burst) if (i_t - t_steps_in_burst) > 0 else 0
                            idx_filter_end = (i_t + 1)

                            filtered_spks_for_e_cells[i_t, i_c] = burst_kernel(spks_for_e_cells[idx_filter_start: idx_filter_end, i_c])


                    # STDP FOR E CELLS: put in pairwise STDP on filtered_spks_for_e_cells
                    stdp_burst_pair = 0

                    for i_t in range(spks_for_e_cells.shape[0]):
                        stdp_start = i_t - m.CUT_IDX_TAU_PAIR if i_t - m.CUT_IDX_TAU_PAIR > 0 else 0

                        stdp_spk_hist = filtered_spks_for_e_cells[stdp_start:i_t, :]

                        t_points_for_stdp = stdp_spk_hist.shape[0]
                        curr_spks = filtered_spks_for_e_cells[i_t, :]

                        if t_points_for_stdp > 0:
                            sparse_curr_spks = csc_matrix(curr_spks)
                            sparse_spks = csc_matrix(np.flip(stdp_spk_hist, axis=0))

                            # compute sparse pairwise correlations with curr_spks and spikes in stdp pairwise time window & dot into pairwise kernel
                            stdp_burst_pair_for_t_step = kron(curr_spks, sparse_spks).T.dot(m.KERNEL_PAIR[:t_points_for_stdp]).reshape(spks_for_e_cells.shape[1], spks_for_e_cells.shape[1])
                            stdp_burst_pair += stdp_burst_pair_for_t_step
                            stdp_burst_pair -= stdp_burst_pair_for_t_step.T
                        else:
                            stdp_burst_pair += 0.


                    # E SINGLE-CELL FIRING RATE RULE
                    fr_update_e = 0

                    e_diffs = e_cell_fr_setpoints - np.sum(spks_for_e_cells > 0, axis=0)
                    e_diffs[e_diffs > 0] = 0
                    fr_update_e = e_diffs.reshape(e_diffs.shape[0], 1) * np.ones((m.N_EXC + m.N_SILENT, m.N_EXC + m.N_SILENT)).astype(float)


                    # E POPULATION-LEVEL FIRING RATE RULE
                    fr_pop_update = e_cell_pop_fr_setpoint - np.sum(spks_for_e_cells)
                    print(m.ETA * m.GAMMA * fr_pop_update)
                    fr_pop_step_mean = m.GAMMA * fr_pop_update
                    fr_pop_step = gaussian((m.N_EXC, m.N_EXC), fr_pop_step_mean, np.abs(fr_pop_step_mean))

                    # I SINGLE-CELL FIRING RATE RULE
                    # fr_update_i = 0

                    # if i_e >= m.I_SINGLE_FR_TRIALS[0] and i_e < m.I_SINGLE_FR_TRIALS[1]:
                    #     if i_cell_fr_setpoints is None:
                    #         i_cell_fr_setpoints = np.sum(spks_for_i_cells > 0, axis=0)
                    #     else:
                    #         i_cell_fr_setpoints += np.sum(spks_for_i_cells > 0, axis=0)
                    # elif i_e == m.I_SINGLE_FR_TRIALS[1]:
                    #     i_cell_fr_setpoints = i_cell_fr_setpoints / (m.I_SINGLE_FR_TRIALS[1] - m.I_SINGLE_FR_TRIALS[0])
                    # elif i_e > m.I_SINGLE_FR_TRIALS[1]:
                    #     i_diffs = i_cell_fr_setpoints - np.sum(spks_for_i_cells > 0, axis=0)
                    #     fr_update_i = i_diffs.reshape(i_diffs.shape[0], 1) * np.ones((m.N_INH, m.N_EXC + m.N_SILENT)).astype(float)


                    e_total_potentiation = m.ETA * (m.ALPHA_1 * fr_update_e + m.BETA * stdp_burst_pair + fr_pop_step)

                    if type(e_total_potentiation) is not float:
                        e_total_potentiation[:m.DROPOUT_MIN_IDX, :] = 0
                        e_total_potentiation[m.DROPOUT_MAX_IDX:, :] = 0 
                        e_total_potentiation[:, :m.DROPOUT_MIN_IDX] = 0
                        e_total_potentiation[:, m.DROPOUT_MAX_IDX:] = 0


                    w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)] += (e_total_potentiation * w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)])
                    w_r_copy['E'][w_r_copy['E'] < 0] = 0

                    # output weight bound
                    cell_outgoing_weight_totals = w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)].sum(axis=0)
                    rescaling = np.where(cell_outgoing_weight_totals > m.CELL_OUTPUT_MAX, m.CELL_OUTPUT_MAX / cell_outgoing_weight_totals, 1.)
                    w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)] *= rescaling.reshape(1, rescaling.shape[0])

                    w_e_e_hard_bound = m.W_E_E_R_MAX / m.PROJECTION_NUM
                    w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)][w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)] > w_e_e_hard_bound] = w_e_e_hard_bound 


                    if i_e % 10 == 0:
                        if i_e < m.DROPOUT_ITER:
                            if i_e % 250 == 0:
                                sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{d_idx}_eidx_{zero_pad(i_e, 4)}', {
                                    'first_spk_times': first_spk_times,
                                    'w_r_e_summed': np.sum(rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC], axis=1),
                                    'w_r_e_i_summed': np.sum(rsp.ntwk.w_r['E'][m.N_EXC:, :m.N_EXC], axis=1),
                                    'w_r_e': rsp.ntwk.w_r['E'],
                                    'w_r_i': rsp.ntwk.w_r['I'],
                                    'spk_bins': spk_bins,
                                    'freqs': freqs,
                                    'exc_raster': exc_raster,
                                    'inh_raster': inh_raster,
                                })
                            else:
                                sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{d_idx}_eidx_{zero_pad(i_e, 4)}', {
                                    'first_spk_times': first_spk_times,
                                    'w_r_e_summed': np.sum(rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC], axis=1),
                                    'w_r_e_i_summed': np.sum(rsp.ntwk.w_r['E'][m.N_EXC:, :m.N_EXC], axis=1),
                                    'spk_bins': spk_bins,
                                    'freqs': freqs,
                                    'exc_raster': exc_raster,
                                    'inh_raster': inh_raster,
                                })
                        else:
                            if i_e % 250 == 0:
                                sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{d_idx}_eidx_{zero_pad(i_e, 4)}', {
                                    'first_spk_times': first_spk_times,
                                    'w_r_e_summed': np.sum(rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC], axis=1),
                                    'w_r_e_i_summed': np.sum(rsp.ntwk.w_r['E'][m.N_EXC:, :m.N_EXC], axis=1),
                                    'w_r_e': rsp.ntwk.w_r['E'],
                                    'w_r_i': rsp.ntwk.w_r['I'],
                                    'spk_bins': spk_bins,
                                    'freqs': freqs,
                                    'exc_cells_initially_active': exc_cells_initially_active,
                                    'exc_cells_newly_active': exc_cells_newly_active,
                                    'inh_raster': inh_raster,
                                    'surviving_cell_indices': surviving_cell_indices,
                                })                               
                            else:
                                sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{d_idx}_eidx_{zero_pad(i_e, 4)}', {
                                    'first_spk_times': first_spk_times,
                                    'w_r_e_summed': np.sum(rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC], axis=1),
                                    'w_r_e_i_summed': np.sum(rsp.ntwk.w_r['E'][m.N_EXC:, :m.N_EXC], axis=1),
                                    'spk_bins': spk_bins,
                                    'freqs': freqs,
                                    'exc_cells_initially_active': exc_cells_initially_active,
                                    'exc_cells_newly_active': exc_cells_newly_active,
                                    'inh_raster': inh_raster,
                                    'surviving_cell_indices': surviving_cell_indices,
                                })
                end = time.time()
                secs_per_cycle = f'{end - start}'
                secs_per_cycle = secs_per_cycle[:secs_per_cycle.find('.') + 2]
                print(f'{secs_per_cycle} s')



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

title = f'{args.title[0]}_ff_{clip(M.W_E_E_R / (0.26 * 0.004))}_eir_{clip(M.W_E_I_R * 1e5)}_ier_{clip(M.W_I_E_R * 1e5)}'

for i in range(1):
    w_r_e = None
    w_r_i = None
    if args.load_mat is not None and args.load_mat[0] is not '':
        w_r_e, w_r_i = load_weight_matrices(args.load_mat[0], 225)

    all_rsps = quick_plot(M, run_title=title, w_r_e=w_r_e, w_r_i=w_r_i, dropouts=[
        {'E': M.DROPOUT_SEV, 'I': 0},
    ])