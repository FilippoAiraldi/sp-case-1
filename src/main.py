import numpy as np
import util
import models
from typing import Dict
import argparse


def compute_EV_and_EEV(pars: Dict[str, np.ndarray], args) -> None:
    # compute the EV solution
    EV_f, EV_sol = models.optimize_EV(pars, verbose=args.verbose,
                                      intvars=args.intvars)

    # print EV solution
    util.print_title('Expected Value Problem')
    print(f'Opt. objective = {EV_f:.3f}')
    for var, value in EV_sol.items():
        print(f'{var} = \n', value)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', type=int, default=int(1e5), 
                        help='Number of LHS samples to approximate the '
                        'recourse model.')
    parser.add_argument('-iv', '--intvars', action='store_true',
                        help='Use integer variables in the optimization; '
                        'otherwise, variables are continuous.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Versobe Gurobi output')
    args = parser.parse_args()

    # set other options
    np.set_printoptions(precision=3, suppress=False)

    # create parameters and LHS samples
    pars = util.get_parameters()
    samples = util.draw_samples(args.samples, pars)

    # run
    compute_EV_and_EEV(pars, args)
