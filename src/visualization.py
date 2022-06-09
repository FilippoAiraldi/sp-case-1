import os
import pickle
import numpy as np
import util
import recourse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from multiprocessing import Pool
from functools import partial


colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#A2142F']


def fig1_computations(pars, EV_vars_C, EV_vars_I, sample_size):
    samples = util.draw_samples(sample_size, pars)
    EEV_C = recourse.optimize_EEV(pars, EV_vars_C, samples)
    EEV_I = recourse.optimize_EEV(pars, EV_vars_I,
                                  samples.astype(int), intvars=True)
    return np.mean(EEV_C), np.mean(EEV_I)


def fig1(pars):
    # solve EV for both continuos and integer variables
    EV_C, EV_vars_C, _ = recourse.optimize_EV(pars, intvars=False)
    EV_I, EV_vars_I, _ = recourse.optimize_EV(pars, intvars=True)

    # solve EEV for different sample sizes and for both continuos and integer
    N = 20  # number of repetitions per size
    S = 50  # number of sizes to try out
    sizes = np.unique(np.logspace(1, 4, S, dtype=int))
    S = sizes.size
    EEV_C, EEV_I = np.empty((S, N)), np.empty((S, N))

    # run parallel computations
    f = partial(fig1_computations, pars, EV_vars_C, EV_vars_I)
    p = Pool()
    with p:
        for i, results in tqdm(enumerate(p.imap(f, np.tile(sizes, N))),
                               total=S * N, desc='Figure 1'):
            n, s = i // S, i % S
            EEV_C[s, n], EEV_I[s, n] = results

    # plot EEV-EV gaps
    fig, ax = plt.subplots(figsize=(7, 2), constrained_layout=True)
    gap_mean = (EEV_C - EV_C).mean(1)
    gap_std = (EEV_C - EV_C).std(1)
    ax.semilogx(sizes, gap_mean, label='continuous',
                color=colors[0], linewidth=1, zorder=4)
    ax.fill_between(sizes, gap_mean + gap_std, gap_mean - gap_std,
                    color=colors[0], alpha=0.2, linewidth=0, zorder=2)

    gap_mean = (EEV_I - EV_I).mean(1)
    gap_std = (EEV_C - EV_C).std(1)
    ax.semilogx(sizes, gap_mean, label='integer',
                color=colors[1], linewidth=1, zorder=3)
    ax.fill_between(sizes, gap_mean + gap_std, gap_mean - gap_std,
                    color=colors[1], alpha=0.2, linewidth=0, zorder=1)

    # make it nicer
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('EEV $-$ EV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.legend(frameon=False)
    fig.savefig('fig1.png', format='png', dpi=300)


def fig2_computations(pars, sample_size):
    samples = util.draw_samples(sample_size, pars)
    TS_C, _, prob_C = recourse.optimize_TS(pars, samples)
    TS_I, _, prob_I = recourse.optimize_TS(pars, samples.astype(int),
                                           intvars=True)
    return TS_C, prob_C, TS_I, prob_I


def fig2(pars):
    # solve TS for different sample sizes and for both continuos and integer
    N = 20  # number of repetitions per size
    S = 50  # number of sizes to try out
    sizes = np.logspace(1, 4, S).astype(int)
    TS_C, prob_C = np.empty((S, N)), np.empty((S, N))
    TS_I, prob_I = np.empty((S, N)), np.empty((S, N))

    # run parallel computations
    f = partial(fig2_computations, pars)
    p = Pool()
    with p:
        for i, results in tqdm(enumerate(p.imap(f, np.tile(sizes, N))),
                               total=S * N, desc='Figure 2'):
            n, s = i // S, i % S
            TS_C[s, n], prob_C[s, n], TS_I[s, n], prob_I[s, n] = results

    # plot TS objs
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    TS_mean, TS_std = TS_C.mean(1), TS_C.std(1)
    axs[0].semilogx(sizes, TS_mean, label='continuous',
                    color=colors[0], linewidth=1, zorder=4)
    axs[0].fill_between(sizes, TS_mean + TS_std, TS_mean - TS_std,
                        color=colors[0], alpha=0.2, linewidth=0, zorder=2)

    TS_mean, TS_std = TS_I.mean(1), TS_I.std(1)
    axs[0].semilogx(sizes, TS_mean, label='integer',
                    color=colors[1], linewidth=1, zorder=3)
    axs[0].fill_between(sizes, TS_mean + TS_std, TS_mean - TS_std,
                        color=colors[1], alpha=0.2, linewidth=0, zorder=1)

    # plot purchase probs
    prob_mean, prob_std = prob_C.mean(1), prob_C.std(1)
    axs[1].semilogx(sizes, prob_mean, label='continuous',
                    color=colors[0], linewidth=1, zorder=4)
    axs[1].fill_between(sizes, prob_mean + prob_std, prob_mean - prob_std,
                        color=colors[0], alpha=0.2, linewidth=0, zorder=2)

    prob_mean, prob_std = prob_I.mean(1) * 100, prob_I.std(1) * 100
    axs[1].semilogx(sizes, prob_mean, label='integer',
                    color=colors[1], linewidth=1, zorder=3)
    axs[1].fill_between(sizes, prob_mean + prob_std, prob_mean - prob_std,
                        color=colors[1], alpha=0.2, linewidth=0, zorder=1)

    # make it nicer
    axs[1].set_xlabel('Sample Size')
    axs[0].set_ylabel('Objective Value')
    axs[1].set_ylabel('Purchase Probability')
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[0].tick_params(which='both', direction='in')
    axs[1].tick_params(which='both', direction='in')
    axs[1].yaxis.set_major_formatter(PercentFormatter())
    axs[0].legend(frameon=False, loc='lower right')
    fig.subplots_adjust(hspace=0.4)
    fig.savefig('fig2.png', format='png', dpi=300)


if __name__ == '__main__':
    # # load all pickle files, for both continuous and integer
    # data = {}
    # for root, _, files in os.walk('results\\recourse'):
    #     for file in files:
    #         if file.endswith('.pkl'):
    #             with open(os.path.join(root, file), 'rb') as f:
    #                 data[file] = pickle.load(f)

    # get constant parameters
    pars = util.get_parameters()

    # do plotting
    fig1(pars)  # ~ 3min
    fig2(pars)  # > 1h
    plt.show()
