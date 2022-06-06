import os
import pickle
# import numpy as np

import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter

from typing import Dict, Any


colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#A2142F']


def figure1_EEV_EV_gap(data: Dict[str, Dict[str, Any]]):
    # from the data, extract tuples (sample size, EEV-EV)
    data_con, data_int = [], []
    for d in data.values():
        args = d['args']
        results = d['results']
        (data_int if args['intvars'] else data_con).append(
            (args['samples'], results['EEV'] - results['EV']['obj']))
    data_con.sort(key=lambda o: o[0])
    data_int.sort(key=lambda o: o[0])

    # plot
    _, ax = plt.subplots(figsize=(7, 2), constrained_layout=True)
    x, y = list(zip(*data_con))
    ax.semilogx(x, y, label='continuous', marker='o', color=colors[0])
    x, y = list(zip(*data_int))
    ax.semilogx(x, y, label='integer', marker='o', color=colors[1])

    # make it nicer
    ax.set_xlabel('Sample size')
    ax.set_ylabel('EEV $-$ EV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="in")
    ax.set_ylim(10, 1e4)
    ax.set_ylim(27.7e3, 31e3)
    ax.legend()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


if __name__ == '__main__':
    # load all pickle files, for both continuous and integer
    data = {}
    for root, _, files in os.walk('results\\recourse'):
        for file in files:
            if file.endswith('.pkl'):
                with open(os.path.join(root, file), 'rb') as f:
                    data[file] = pickle.load(f)

    figure1_EEV_EV_gap(data)

    plt.show()
