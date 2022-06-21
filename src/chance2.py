import gurobipy as gb
from gurobipy import GRB
import numpy as np
import scipy
import util
import recourse
from itertools import product


def draw_new_samples(args, pars):
    n, s = pars['n'], pars['s']
    p = (n * s) // 2

    # draw demand samples
    demands = util.draw_samples(args.samples, pars, seed=args.seed,
                                asint=args.intvars).reshape(-1, 2 * p)

    # translate the samples to the new distribution
    Nblock = np.tril(np.ones((p, p)))
    N = scipy.linalg.block_diag(Nblock, Nblock)
    new_demands = (N @ demands.T).T

    # also compute the new mean and new covariance
    new_mean = N @ pars['demand_mean'].reshape(-1, 1)
    new_cov = N @ np.diag(np.square(pars['demand_std'].flatten())) @ N.T
    return new_demands, new_mean, new_cov


def run_LP_11(samples, pars):
    # unpack the samples into demands, mean and cov
    demands, demand_mean, demand_cov = samples

    # define some constants
    N, p = demands.shape    # number of scenarios, size of demand vector (p=8)
    alpha = 0.5             # alpha
    M = np.ones(N) * 1e4    # a large number

    # create a model
    mdl = gb.Model(name='CC')

    # create 1st stage variables and 2nd stage variables, one per scenario
    vars1 = recourse.add_1st_stage_variables(mdl, pars, intvars=args.intvars)
    vars2 = recourse.add_2nd_stage_variables(mdl, pars, intvars=args.intvars)

    # create binary variables
    delta = np.array(mdl.addVars(N, vtype=GRB.BINARY, name='delta').values())

    # add each constraint in the order in which they are written
    Ym = vars2['Y-'].reshape(-1, 1)

    # add constraint 1
    cons = [(Ym[i] + M[i] * delta[j] - demands[j, i]).item()
            for j, i in product(range(N), range(p))]
    mdl.addConstrs((cons[i] >= 0 for i in range(len(cons))), name='con_1')

    # add constraint 2
    mdl.addConstr(1 / N * delta.sum() <= 1 - alpha, name='con_2')

    # add constraint 3
    # ... TODO ...

    # add constraint 4
    recourse.add_1st_stage_constraints(mdl, pars, vars1)

    # add constraint 5
    sigma_bar = np.sqrt(np.diag(demand_cov))
    con = (Ym - demand_mean + 3 * sigma_bar).flatten()
    mdl.addConstrs((con[i] >= 0 for i in range(con.size)), name='con_5')

    # solve
    mdl.optimize()

    # if model is infeasible, return none
    if mdl.Status == GRB.INFEASIBLE:
        mdl.dispose()
        return None

    # extract the numeric values
    objval = mdl.ObjVal
    convert = ((lambda o: util.var2val(o).astype(int))
               if args.intvars else
               (lambda o: util.var2val(o)))
    sol1 = {name: convert(var) for name, var in vars1.items()}
    sol2 = {name: convert(var) for name, var in vars2.items()}
    delta = convert(delta).astype(int)

    # return solution objective and values
    mdl.dispose()
    return objval, sol1, sol2, delta


# this stuff should be in the main.py
np.set_printoptions(precision=3, suppress=False)
args = util.parse_args()
pars = util.get_parameters()
demands = draw_new_samples(args, pars)
run_LP_11(demands, pars)
