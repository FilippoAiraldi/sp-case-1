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


def fig1_computations(pars, EV_vars_C, EV_vars_I, samples):
    EEV_C = recourse.optimize_EEV(pars, EV_vars_C, samples)
    EEV_I = recourse.optimize_EEV(pars, EV_vars_I,
                                  samples.astype(int), intvars=True)
    return np.mean(EEV_C), np.mean(EEV_I)


def fig1(args, pars):
    # solve EV for both continuos and integer variables
    EV_C, EV_vars_C, _ = recourse.optimize_EV(pars, intvars=False)
    EV_I, EV_vars_I, _ = recourse.optimize_EV(pars, intvars=True)

    # solve EEV for different sample sizes and for both continuos and integer
    N = 500
    sizes = np.logspace(1, 4, N).astype(int)
    samples = [util.draw_samples(size, pars, seed=args.seed) for size in sizes]
    EEV_C, EEV_I = [], []
    p = Pool()
    f = partial(fig1_computations, pars, EV_vars_C, EV_vars_I)
    with p:
        for results in tqdm(p.imap(f, samples), total=N, desc='Figure 1'):
            EEV_C.append(results[0])
            EEV_I.append(results[1])

    # convert to arrays
    EEV_C, EEV_I = np.array(EEV_C), np.array(EEV_I)

    # plot EEV-EV gaps
    _, ax = plt.subplots(figsize=(7, 2), constrained_layout=True)
    ax.semilogx(sizes, EEV_C - EV_C, label='continuous',
                color=colors[0], linewidth=1, zorder=3)
    ax.semilogx(sizes, EEV_I - EV_I, label='integer',
                color=colors[1], linewidth=1, zorder=2)

    # make it nicer
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('EEV $-$ EV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')
    ax.legend(loc='lower right', frameon=False)


def fig2_computations(pars, samples):
    TS_C, _, prob_C = recourse.optimize_TS(pars, samples)
    TS_I, _, prob_I = recourse.optimize_TS(pars, samples.astype(int),
                                           intvars=True)
    return TS_C, prob_C, TS_I, prob_I


def fig2(args, pars):
    # solve TS for different sample sizes and for both continuos and integer
    N = 100
    sizes = np.logspace(1, 4, N).astype(int)
    samples = [util.draw_samples(size, pars, seed=args.seed) for size in sizes]
    TS_C, TS_I, prob_C, prob_I = [], [], [], []
    p = Pool()
    f = partial(fig2_computations, pars)
    with p:
        for results in tqdm(p.imap(f, samples), total=N, desc='Figure 2'):
            TS_C.append(results[0])
            prob_C.append(results[1] * 100)
            TS_I.append(results[2])
            prob_I.append(results[3] * 100)

    # plot EEV-EV gaps
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True,
                            constrained_layout=True)
    axs[0].semilogx(sizes, TS_C, label='continuous',
                    color=colors[0], linewidth=1, zorder=3)
    axs[0].semilogx(sizes, TS_I, label='integer',
                    color=colors[1], linewidth=1, zorder=2)
    axs[1].semilogx(sizes, prob_C, label='continuous',
                    color=colors[0], linewidth=1, zorder=3)
    axs[1].semilogx(sizes, prob_I, label='integer',
                    color=colors[1], linewidth=1, zorder=2)

    # make it nicer
    axs[1].set_xlabel('Sample Size')
    axs[0].set_ylabel('Objective Value')
    axs[1].set_ylabel('Purchase Probability')
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[0].tick_params(direction='in')
    axs[1].tick_params(direction='in')
    axs[1].yaxis.set_major_formatter(PercentFormatter())
    axs[0].legend(loc='lower right', frameon=False)
    fig.subplots_adjust(hspace=0.4)


if __name__ == '__main__':
    # # load all pickle files, for both continuous and integer
    # data = {}
    # for root, _, files in os.walk('results\\recourse'):
    #     for file in files:
    #         if file.endswith('.pkl'):
    #             with open(os.path.join(root, file), 'rb') as f:
    #                 data[file] = pickle.load(f)

    # parse arguments and get constant parameters
    args = util.parse_args()
    args.verbose = 0
    pars = util.get_parameters()

    # do plotting
    # fig1(args, pars)
    # fig2(args, pars)
    plt.show()
