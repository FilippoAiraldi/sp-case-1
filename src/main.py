import numpy as np
import util
import models
from typing import Dict
import argparse


def point_a(pars: Dict[str, np.ndarray],
            samples: np.ndarray, args) -> None:
    '''Runs computations for points a (EV, EEV and TS).'''

    import time
    starttime = time.process_time()

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

    # assess TS quality via MRP
    CI = models.run_MRP(pars, TS_vars1, samples.shape[0], alpha=args.alpha,
                        replicas=args.replicas, intvars=args.intvars,
                        verbose=args.verbose)
    print('Bound on optimality gap =', CI)

    # compute WS
    util.print_title('Wait-and-See')

    WS_objs = models.optimize_WS(
        pars, samples, intvars=args.intvars, verbose=args.verbose)
    WS_obj = np.mean(WS_objs)

    print(f'WS = {WS_obj:.3f}',
          f'({samples.shape[0] - len(WS_objs)} scenarios were infeasible)')

    # compute VSS and EVPI
    VSS = EEV_obj - TS_obj
    EVPI = TS_obj - WS_obj
    print('VSS =', VSS, '- EVPI =', EVPI)

    import pickle
    intvars = 'I'if args.intvars else 'C'
    print('HEEEEEY', args.__dict__)
    with open(f'a_samples_{samples.shape[0]}_type_{intvars}.pkl', 'wb') as f:
        pickle.dump({
            'execution time': time.process_time() - starttime,
            'args': args.__dict__,
            'EV': {'obj': EV_obj, 'sol': EV_vars1 | EV_vars2},
            'EEV': EEV_obj,
            'TS': {'obj': TS_obj, 'sol': TS_vars1},
            'MRP_CI': CI,
            'WS': WS_obj,
            'VSS': VSS,
            'EVPI': EVPI
        }, f)
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
    parser.add_argument('-a', '--alpha', type=float, default=0.95,
                        metavar='(0-1)', help='MRP Confidence level.')
    parser.add_argument('-r', '--replicas', type=int, default=30,
                        help='MRP number of replicas.')
    parser.add_argument('-iv', '--intvars', action='store_true',
                        help='Use integer variables in the optimization; '
                        'otherwise, variables are continuous.')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed.')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        choices=[0, 1, 2], help='Verbosity of Gurobi output')
    args = parser.parse_args()
    ok = args.samples > 0 and 0 < args.alpha < 1 and args.replicas > 0
    assert ok, 'invalid arguments'

    # set other options
    np.set_printoptions(precision=3, suppress=False)

    # create parameters and LHS samples
    constant_pars = util.get_parameters()

    # samples to approximate the continuous demands distributions as discrete
    samples = util.draw_samples(args.samples, constant_pars,
                                asint=args.intvars, seed=args.seed)

    # run point (a)
    point_a(pars=constant_pars, samples=samples, args=args)
