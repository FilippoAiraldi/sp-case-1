import numpy as np
from typing import Dict


def get_parameters() -> Dict[str, np.ndarray]:
    '''
    Gets the problem constant parameters.

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
