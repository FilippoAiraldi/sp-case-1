import numpy as np
from scipy.stats import qmc
from scipy.stats.distributions import norm
from itertools import product
from typing import Dict


def get_parameters() -> Dict[str, np.ndarray]:
    '''
    Gets the constant parameters for the problem.

    Returns
    -------
    pars :  dict[str, np.ndarray]
        Dictionary containing the problem parameters.
    '''
    s = 4
    return {
        # nb of periods t (t = 1,...,s)
        's': s,

        # nb of products i (i = 1,...,n)
        'n': 2,

        # nb of sources j (j = 1,...,m)
        'm': 3,

        # mean and std of demands for product i at time t (i,t)
        'demand_mean': np.array([[30, 32, 44, 48],
                                 [50, 50, 50, 60]]) * 10,
        'demand_std': np.tile(np.array([[45], [75]]), (1, s)),

        # units of source j needed per unit of product i (i,j)
        'A': np.array([[4, 3, 3],
                       [5, 2, 7]]),

        # units of source j available at time t (j,t)
        'B': np.array([[40, 40, 40, 35],
                       [30, 30, 25, 30],
                       [45, 45, 37.5, 35]]) * 100,

        # upper bound on extra capacity of source j at time t (j,t)
        'UB': np.array([[40, 40, 40, 35],
                        [30, 30, 25, 35],
                        [45, 45, 37.5, 35]]) * 10,

        # production cost of product i at time t (i,t)
        'C1': np.tile(np.array([[100], [150]]), (1, s)),

        # extra capacity cost for source j at timet t (j,t)
        'C2': np.tile(np.array([[15], [20], [10]]), (1, s)),

        # workforce increase/decrease cost from time t-1 to t (t-1)
        'C3+': np.full(s - 1, 20),  # increase
        'C3-': np.full(s - 1, 15),  # decrease

        # unit cost for shortage/surplus of product i at time t (i,t)
        'Q+': np.tile(np.array([[400], [450]]), (1, s)),    # shortage
        'Q-': np.array([[25, 25, 25, 100],                  # surplus
                        [30, 30, 30, 150]]),
    }


def draw_samples(nb_samples: int, pars: Dict[str, np.ndarray]) -> np.ndarray:
    '''
    Draws normally distributed samples via LHS.

    Parameters
    ----------
    nb_samples : int
        Number of samples to draw.
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.

    Returns
    -------
    samples : np.ndarray
        A nb_samples-by-n-by-s matrix, where each sample contains a normally 
        distributed subsample of demand per product per period
    '''
    # one sample contains one demand for each product for each period, meaning
    # that each sample an n-by-s matrix
    n, s = pars['n'], pars['s']
    means, stds = pars['demand_mean'], pars['demand_std']

    # collect samples from an uniformely distribution
    sampler = qmc.LatinHypercube(d=n * s)
    samples = sampler.random(n=nb_samples).reshape(-1, n, s)

    # transform to normal distributions
    for i, t in product(range(n), range(s)):
        samples[:, i, t] = \
            norm(loc=means[i, t], scale=stds[i, t]).ppf(samples[:, i, t])
    return samples


def print_title(title: str) -> None:
    '''Prints a nicely formatted title to console.'''
    L = 80
    L1 = (L - len(title)) // 2 - 1
    L2 = L - L1 - len(title) - 2
    print(
        '# ' + '=' * L + ' #',
        '# ' + '=' * L1 + ' ' + title + ' ' + '=' * L2 + ' #',
        '# ' + '=' * L + ' #',
        sep='\n')
