import numpy as np
import util
import recourse
import time
from typing import Dict, Any


def run_recourse(pars: Dict[str, np.ndarray], args) -> Dict[str, Any]:
    '''Runs computations.'''
    # draw samples to approximate continuous demand distributions as discrete
    samples = util.draw_samples(args.samples, pars,
                                asint=args.intvars, seed=args.seed)

    # compute EV
    util.print_title('Expected Value')
    EV_obj, EV_vars1, EV_vars2 = recourse.optimize_EV(
        pars, intvars=args.intvars, verbose=args.verbose)
    print(f'EV = {EV_obj:.3f}')
    for var, value in (EV_vars1 | EV_vars2).items():
        print(f'{var} = \n', value)

    # compute EEV
    util.print_title('Expected result of Expected Value')
    EEV_objs = recourse.optimize_EEV(
        pars, EV_vars1, samples, intvars=args.intvars, verbose=args.verbose)
    EEV_obj = np.mean(EEV_objs)
    print(f'EEV = {EEV_obj:.3f}',
          f'({samples.shape[0] - len(EEV_objs)} scenarios were infeasible)')

    # compute TS
    util.print_title('Two-stage Model')
    TS_obj, TS_vars1, purchase_prob = recourse.optimize_TS(
        pars, samples, intvars=args.intvars, verbose=args.verbose)
    print(f'TS = {TS_obj:.3f} / purchase prob = {purchase_prob}')
    for var, value in TS_vars1.items():
        print(f'{var} = \n', value)

    # assess TS quality via MRP
    CI = recourse.run_MRP(pars, TS_vars1, sample_size=args.samples,
                          alpha=args.alpha, replicas=args.replicas,
                          intvars=args.intvars, verbose=args.verbose,
                          seed=args.seed)
    print('Bound on optimality gap =', CI)

    # compute WS
    util.print_title('Wait-and-See')
    WS_objs = recourse.optimize_WS(
        pars, samples, intvars=args.intvars, verbose=args.verbose)
    WS_obj = np.mean(WS_objs)
    print(f'WS = {WS_obj:.3f}',
          f'({samples.shape[0] - len(WS_objs)} scenarios were infeasible)')

    # compute VSS and EVPI
    VSS = EEV_obj - TS_obj
    EVPI = TS_obj - WS_obj
    print('VSS =', VSS, '- EVPI =', EVPI)

    # compute Sensitivity Analysis w.r.t. labour increase upper bound and cost
    util.print_title('Sensitivity Analysis')
    labor_sens, labor_sens_mtx = recourse.labor_sensitivity_analysis(
        pars, samples, args.lab_factors, intvars=args.intvars,
        verbose=args.verbose)
    print(f'labour sensitivity =\n{labor_sens_mtx}')
    dp_sens, dp_sens_mtx = recourse.dep_sensitivity_analysis(
        pars, args.samples, args.dem_factors, intvars=args.intvars,
        verbose=args.verbose, seed=args.seed)
    print(f'demand sensitivity =\n{dp_sens_mtx}')

    # return the results
    return {
        'EV': {'obj': EV_obj, 'sol': EV_vars1 | EV_vars2},
        'EEV': EEV_obj,
        'TS': {'obj': TS_obj, 'sol': TS_vars1},
        'MRP_CI': CI,
        'WS': WS_obj,
        'VSS': VSS,
        'EVPI': EVPI,
        'purchase probability': purchase_prob,
        'labor sensitivity': {str(k): v for k, v in labor_sens.items()},
        'demand sensitivity': {str(k): v for k, v in dp_sens.items()}
    }


if __name__ == '__main__':
    # initialize
    np.set_printoptions(precision=3, suppress=False)
    args = util.parse_args()
    constant_pars = util.get_parameters()

    # run points
    starttime = time.process_time()
    results = run_recourse(pars=constant_pars, args=args)
    # ...run_chance...

    # save results
    util.save_results(execution_time=time.process_time() - starttime,
                      args=args.__dict__, results=results)
