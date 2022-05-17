import numpy as np
import util
import models
from typing import Dict
import argparse


def compute_EV_and_EEV(pars: Dict[str, np.ndarray], args) -> None:
    # compute the EV solution
    EV_f, EV_sol = models.optimize_EV_solution(pars,
                                               verbose=args.verbose,
                                               intvars=args.intvars)

    # print EV solution
    util.print_title('Expected Value Problems')
    print(f'Opt. objective = {EV_f:.3f}')
    for var, value in EV_sol.items():
        print(f'{var} = \n', value)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-iv', '--intvars', action='store_true',
                        help='Use integer variables in the optimization; '
                        'otherwise, they are continuous')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Versobe Gurobi output')
    args = parser.parse_args()

    # set other options
    np.set_printoptions(precision=3, suppress=False)
    pars = util.get_parameters()

    # run
    compute_EV_and_EEV(pars, args)
