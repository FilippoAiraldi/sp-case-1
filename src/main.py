import numpy as np
import util
import models
from typing import Dict
import argparse


def compute_EV_and_EEV(pars: Dict[str, np.ndarray],
                       samples: np.ndarray, args) -> None:
    # compute the EV solution
    EV_f, EV_sol = models.optimize_EV(pars, intvars=args.intvars,
                                      verbose=args.verbose)

    # print EV solution
    util.print_title('Expected Value')
    print(f'EV = {EV_f:.3f}')
    for var, value in EV_sol.items():
        print(f'{var} = \n', value)

    # compute EEV solution
    EEV_fs = models.optimize_EEV(pars, EV_sol, samples, intvars=args.intvars,
                                 verbose=args.verbose)

    util.print_title('Expected result of Expected Value')
    print(f'EEV = {np.mean(EEV_fs):.3f}',
          f'({samples.shape[0] - len(EEV_fs)} scenarios were infeasible)')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', type=int, default=int(1e5),
                        help='Number of LHS samples to approximate the '
                        'recourse model.')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed.')
    parser.add_argument('-iv', '--intvars', action='store_true',
                        help='Use integer variables in the optimization; '
                        'otherwise, variables are continuous.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Versobe Gurobi output')
    args = parser.parse_args()

    # set other options
    np.set_printoptions(precision=3, suppress=False)

    # create parameters and LHS samples
    constant_pars = util.get_parameters()
    demand_samples = util.draw_samples(args.samples, constant_pars, 
                                       seed=args.seed)

    # run
    compute_EV_and_EEV(pars=constant_pars, samples=demand_samples, args=args)
