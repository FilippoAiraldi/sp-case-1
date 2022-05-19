import numpy as np
import util
import models
from typing import Dict
import argparse


def point_a(pars: Dict[str, np.ndarray],
            samples: np.ndarray, args) -> None:
    '''Runs computations for points a (EV, EEV and TS).'''
    # compute the EV solution
    util.print_title('Expected Value')
    EV_obj, EV_vars1, EV_vars2 = models.optimize_EV(
        pars, intvars=args.intvars, verbose=args.verbose)
    print(f'EV = {EV_obj:.3f}')
    for var, value in (EV_vars1 | EV_vars2).items():
        print(f'{var} = \n', value)

    # compute EEV solution
    util.print_title('Expected result of Expected Value')
    EEV_fs = models.optimize_EEV(
        pars, EV_vars1, samples, intvars=args.intvars, verbose=args.verbose)
    print(f'EEV = {np.mean(EEV_fs):.3f}',
          f'({samples.shape[0] - len(EEV_fs)} scenarios were infeasible)')

    # TODO: solve TS by L-shape?
    # ...


def point_b(): pass
def point_c(): pass
def point_d(): pass
def point_e(): pass


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
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        choices=[0, 1, 2], help='Verbosity of Gurobi output')
    args = parser.parse_args()

    # set other options
    np.set_printoptions(precision=3, suppress=False)

    # create parameters and LHS samples
    constant_pars = util.get_parameters()
    demand_samples = util.draw_samples(args.samples, constant_pars,
                                       seed=args.seed)

    # run point (a)
    point_a(pars=constant_pars, samples=demand_samples, args=args)
