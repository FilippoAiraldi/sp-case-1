import numpy as np
import util
import models
from typing import Dict
import argparse


def point_a(pars: Dict[str, np.ndarray],
            samples: np.ndarray, args) -> None:
    '''Runs computations for points a (EV, EEV and TS).'''

    # compute EV
    util.print_title('Expected Value')

    EV_obj, EV_vars1, EV_vars2 = models.optimize_EV(
        pars, intvars=args.intvars, verbose=args.verbose)

    print(f'EV = {EV_obj:.3f}')
    for var, value in (EV_vars1 | EV_vars2).items():
        print(f'{var} = \n', value)

    # compute EEV
    util.print_title('Expected result of Expected Value')

    EEV_objs = models.optimize_EEV(
        pars, EV_vars1, samples, intvars=args.intvars, verbose=args.verbose)
    EEV_obj = np.mean(EEV_objs)

    print(f'EEV = {EEV_obj:.3f}',
          f'({samples.shape[0] - len(EEV_objs)} scenarios were infeasible)')

    # compute TS
    util.print_title('Two-stage Model')

    TS_obj, TS_vars1 = models.optimize_TS(pars, samples,
                                          intvars=args.intvars,
                                          verbose=args.verbose)

    print(f'EV = {TS_obj:.3f}')
    for var, value in TS_vars1.items():
        print(f'{var} = \n', value)

    # assess TS solution quality via MRP
    # ...

    # compute VSS
    VSS = EEV_obj - TS_obj

    # compute WS
    util.print_title('Wait-and-See')

    WS_objs = models.optimize_WS(
        pars, samples, intvars=args.intvars, verbose=args.verbose)
    WS_obj = np.mean(WS_objs)

    print(f'EEV = {WS_obj:.3f}',
          f'({samples.shape[0] - len(WS_objs)} scenarios were infeasible)')

    return


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

    # samples to approximate the continuous demands distributions as discrete
    # (used in EEV and TS)
    samples = util.draw_samples(args.samples, constant_pars, seed=args.seed)
    if args.intvars:
        samples = samples.astype(int)

    # run point (a)
    point_a(pars=constant_pars, samples=samples, args=args)
