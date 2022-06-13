import os
import pickle
import numpy as np
import util
import recourse
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
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


def fig3():
    # load data for both continuous and integer
    data = {'labor': {}, 'demand': {}}
    for t in ('C', 'I'):
        # load from pickle
        fn = os.path.join('results', 'recourse', f'Samples_10000_Type_{t}.pkl')
        with open(fn, 'rb') as f:
            pkl = pickle.load(f)

        # convert sensitivity to array, normalize and save to dict
        for name in ('labor', 'demand'):
            f = np.array(pkl['args'][f'{name[:3]}_factors'])
            S = np.array(list(pkl['results'][f'{name} sensitivity'].values()))
            S = S.reshape(f.size, f.size)
            data[name][t] = (f, S)

    # for labor - 1st factor (row, y) is extra labor upper bound and
    # 2nd factor (column, x) is extra labor cost
    # for demands - 1st factor (row, y) is inter-product correlation and
    # 2nd factor (column, x) is inter-period correlation
    baseline = (1, 0)
    labels = [('$up_{3t}$ factor', '$c^2_{3t}$ factor'),
              ('inter-product factor', 'inter-period factor')]
    titles = [('labor parameters\nsensitivity'),
              ('demand dependence\nsensitivity')]

    # create figures for labor and demand sensitivities
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # plot labour and demand sensitivity matrices
    for i, (vals, ax) in enumerate(zip(data.values(), axs)):
        f, S = vals['C']
        im = ax.imshow(S)
        ax.set_xticks(np.arange(f.size), labels=f)
        ax.set_yticks(np.arange(f.size), labels=f)

        # make it nicer
        ax.set_ylabel(labels[i][0], fontsize=15)
        ax.set_title(labels[i][1], fontsize=15)  # x label
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        ax.set_xlabel(titles[i], labelpad=17, fontsize=16)  # title
        ax.tick_params(axis='both', which='major', labelsize=14)

        # add colorbar and adjust layout
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=14)
        fig.subplots_adjust(wspace=0.5)

        # add percentage text
        b = np.where(f == baseline[i])[0]
        Snorm = ((S - S[b[0], b[0]]) / S[b[0], b[0]] * 100
                 if b.size == 1 else
                 S)
        for i, j in product(range(S.shape[0]), range(S.shape[1])):
            ax.text(j, i, f'{Snorm[i, j]:+1.1f}%', color='white', fontsize=13,
                    horizontalalignment='center', verticalalignment='center')

    # save figure
    # fig.savefig('fig3.eps', format='eps', dpi=300)


if __name__ == '__main__':
    # get constant parameters
    pars = util.get_parameters()

    # do plotting
    # fig1(pars)  # ~ 3min
    # fig2(pars)  # > 1h
    fig3()  # seconds
    plt.show()
