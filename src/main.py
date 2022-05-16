import numpy as np
import util
import models


def run():
    np.set_printoptions(precision=3)

    # compute the EV solution
    pars = util.get_parameters()
    ev_f, ev_sol = models.optimize_expected_value_solution(pars)
    print('_' * 60)
    print(f'EV solution = {ev_f:.3f}')
    for var, value in ev_sol.items():
        print(f'{var} = \n', value)


if __name__ == '__main__':
    TODO: PARSE ARGUMENT AND WHICH POINT TO SOLVE (i, ii, iii, etc...)
    # parse arguments

    # run 
    run()
