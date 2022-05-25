import numpy as np
from scipy import stats
import gurobipy as gb
from gurobipy import GRB
from itertools import product
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Dict, Tuple, List, Optional
import util


def add_1st_stage_variables(mdl: gb.Model,
                            pars: Dict[str, np.ndarray],
                            intvars: bool = False) -> np.ndarray:
    '''
    Creates the 1st stage variables for the current Gurobi model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model for which to create the 1st stage variables.
    pars : dict[str, int | np.ndarray]
        Dictionary containing the optimization problem parameters.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.

    Returns
    -------
    vars : np.ndarray of gurobipy.Var
        Array containing the 1st stage variables.
    '''
    TYPE = GRB.INTEGER if intvars else GRB.CONTINUOUS
    s, n, m = pars['s'], pars['n'], pars['m']

    vars_ = [('X', (n, s)), ('U', (m, s)), ('Z+', (s - 1,)), ('Z-', (s - 1,))]
    return {
        name: np.array(
            mdl.addVars(*size, lb=0, vtype=TYPE, name=name).values()
        ).reshape(*size) for name, size in vars_
    }


def add_2nd_stage_variables(mdl: gb.Model,
                            pars: Dict[str, np.ndarray],
                            scenarios: int = 1,
                            intvars: bool = False) -> np.ndarray:
    '''
    Creates the 2nd stage variables for the current Gurobi model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model for which to create the 2nd stage variables.
    pars : dict[str, int | np.ndarray]
        Dictionary containing the optimization problem parameters.
    scenarios : int, optional
        If given, creates a set of 2nd stage variables per scenario. Otherwise,
        it defaults to 1.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.

    Returns
    -------
    vars : np.ndarray of gurobipy.Var
        Array containing the 2nd stage variables.
    '''
    TYPE = GRB.INTEGER if intvars else GRB.CONTINUOUS
    s, n = pars['s'], pars['n']
    size = (n, s) if scenarios == 1 else (scenarios, n, s)

    varnames = ('Y+', 'Y-')
    return {
        name: np.array(
            mdl.addVars(*size, lb=0, vtype=TYPE, name=name).values()
        ).reshape(*size) for name in varnames
    }


def fix_var(mdl: gb.Model, var: np.ndarray, value: np.ndarray) -> None:
    '''
    Fixes a variable to a given value. Not recommended to use this in a loop.

    Parameters
    ----------
    mdl : gurobipy.Model
        The model which the variables belong to.
    var : np.ndarray of gurobipy.Var
        The variable to be fixed.
    value : np.ndarray
        The value at which to fix the variable.
    '''
    var = var.flatten().tolist()
    value = value.flatten().tolist()
    mdl.setAttr(GRB.Attr.LB, var, value)
    mdl.setAttr(GRB.Attr.UB, var, value)


def get_1st_stage_objective(pars: Dict[str, np.ndarray],
                            vars: Dict[str, np.ndarray]
                            ) -> Tuple[Union[gb.LinExpr, float], ...]:
    '''
    Computes the 1st stage objective.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, np.ndarray]
        1st stage variables to use to compute the objective. The arrays can be 
        either symbolical, or numerical.

    Returns
    -------
    obj :  gurobipy.LinExpr | float
        An expression (if variables are symbolical) or a number (if vars are
        umerical) representing the 1st stage objective.
    '''
    vars_ = [vars['X'], vars['U'], vars['Z+'], vars['Z-']]
    costs = [pars['C1'], pars['C2'], pars['C3+'], pars['C3-']]
    return gb.quicksum((cost * var).sum() for var, cost in zip(vars_, costs))


def get_2nd_stage_objective(pars: Dict[str, np.ndarray],
                            vars: Dict[str, np.ndarray]
                            ) -> Tuple[Union[gb.LinExpr, float], ...]:
    '''
    Computes the 2nd stage objective.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, np.ndarray]
        2nd stage variables to use to compute the objective. The arrays can be 
        either symbolical, or numerical. If the arrays have 3 dimensions, then
        the first dimension is regarded as the number of scenarios.

    Returns
    -------
    obj :  gurobipy.LinExpr | float
        An expression (if variables are symbolical) or a number (if vars are
        umerical) representing the 2nd stage objective.
    '''
    # get the number of scenarios (if 2D, then 1; if 3D, then first dimension)
    S = 1 if vars['Y+'].ndim == 2 else vars['Y+'].shape[0]

    # get variables and costs
    vars_ = [vars['Y+'], vars['Y-']]
    costs = [pars['Q+'], pars['Q-']]

    obj = gb.quicksum((cost * var).sum() for var, cost in zip(vars_, costs))
    return obj / S


def add_1st_stage_constraints(mdl: gb.Model,
                              pars: Dict[str, np.ndarray],
                              vars: Dict[str, np.ndarray]) -> None:
    '''
    Adds 1st stage constraints to the model.

    Parameters
    ----------
    mdl : gurobipy
        Model to add constraints to.
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, np.ndarray]
        Dictionary containing the optimization variables.
    '''
    X, U, Zp, Zm = vars['X'], vars['U'], vars['Z+'], vars['Z-']

    # sufficient sources to produce necessary products
    A, B = pars['A'], pars['B']
    con = (A.T @ X - B - U).flatten()
    mdl.addConstrs((con[i] <= 0 for i in range(con.size)), name='con_product')

    # work force level increase/decrease
    con = Zp - Zm - A[:, -1] @ (X[:, 1:] - X[:, :-1])
    mdl.addConstrs((con[i] == 0 for i in range(con.size)), name='con_work')

    # extra capacity upper bounds
    UB = pars['UB']
    con = (U - UB).flatten()
    mdl.addConstrs((con[i] <= 0 for i in range(con.size)), name='con_extracap')


def add_2nd_stage_constraints(mdl: gb.Model,
                              pars: Dict[str, np.ndarray],
                              vars_1st: Dict[str, np.ndarray],
                              vars_2nd: Dict[str, np.ndarray],
                              demands: np.ndarray = None
                              ) -> Optional[gb.MVar]:
    '''
    Adds 2nd stage constraints with some deterministic values in place of 
    random demand variables to the model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model to add constraints to.
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars_1st : dict[str, np.ndarray]
        Dictionary containing the 1st stage optimization variables.
    vars_2nd : dict[str, np.ndarray]
        Dictionary containing the 2nd stage optimization variables.
    demands : np.ndarray, optional
        Deterministic demand values. If None, new variables are added to the 
        model and returned.

    Returns
    -------
    demands : np.ndarray of gurobipy.MVar
        The new demand variables used in the constraints. Only created and 
        returned when no demand is passed in the arguments.
    '''
    # get the number of scenarios (if 2D, then 1; if 3D, then first dimension)
    S = 1 if vars_2nd['Y+'].ndim == 2 else vars_2nd['Y+'].shape[0]

    s, n = pars['s'], pars['n']
    X, Yp, Ym = vars_1st['X'], vars_2nd['Y+'], vars_2nd['Y-']

    if demands is None:
        return_ = True
        size = (n, s) if S == 1 else (S, n, s)
        demands = np.array(
            mdl.addVars(*size, lb=0, ub=0, name='demand').values()
        ).reshape(size)
    else:
        return_ = False

    # in the first period, zero surplus is assumed
    Ym = np.concatenate((np.zeros((*Ym.shape[:-1], 1)), Ym), axis=-1)
    con = (X + Ym[..., :-1] + Yp - Ym[..., 1:] - demands).flatten()
    mdl.addConstrs((con[i] == 0 for i in range(con.size)), name='con_demand')

    if return_:
        return demands


def optimize_EV(
    pars: Dict[str, np.ndarray],
    intvars: bool = False,
    verbose: int = 0
) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    '''
    Computes the Expected Value solution via a Gurobi model.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.

    Returns
    -------
    obj : float
        Value of the objective function at the optimal point.
    solution : dict[str, np.ndarray]
        Dictionary containing the value of each variable at the optimum.
    '''
    # initialize model
    mdl = gb.Model(name='EV')
    if verbose < 1:
        mdl.Params.LogToConsole = 0

    # create the variables
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)
    vars2 = add_2nd_stage_variables(mdl, pars, intvars=intvars)

    # get the 1st and 2nd stage objectives, and sum them up
    mdl.setObjective(get_1st_stage_objective(pars, vars1) +
                     get_2nd_stage_objective(pars, vars2), GRB.MINIMIZE)

    # add 1st stage constraints
    add_1st_stage_constraints(mdl, pars, vars1)

    # to get the EV Solution, in the second stage constraints the random
    # variables are replaced with their expected value
    d_mean = pars['demand_mean']
    if intvars:
        d_mean = d_mean.astype(int)
    add_2nd_stage_constraints(mdl, pars, vars1, vars2, demands=d_mean)

    # run optimization
    mdl.optimize()

    # retrieve optimal objective value and optimal variables
    objval = mdl.ObjVal
    convert = ((lambda o: util.var2val(o).astype(int))
               if intvars else
               (lambda o: util.var2val(o)))
    sol1 = {name: convert(var) for name, var in vars1.items()}
    sol2 = {name: convert(var) for name, var in vars2.items()}
    mdl.dispose()
    return objval, sol1, sol2


def optimize_EEV(pars: Dict[str, np.ndarray],
                 EV_vars1: Dict[str, np.ndarray],
                 samples: np.ndarray,
                 intvars: bool = False,
                 verbose: int = 0) -> List[float]:
    '''
    Computes the expected value of the Expected Value solution via multiple 
    Gurobi models.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    EV_vars1 : dict[str, np.ndarry]
        Dictionary containing the solution to the EV problem.
    samples : np.ndarray
        Array of different samples/scenarios approximating the demand 
        distributions.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.

    Returns
    -------
    objs : list[float]
        A list containing the optimal objective for each scenario.
    '''
    # create a starting model for the first scenario. Instead of instantiating
    # a new one, the following scenarios will use it again
    mdl = gb.Model(name='EEV')
    if verbose < 2:
        mdl.Params.LogToConsole = 0

    # create only 2nd variables, 1st stage variables are taken from the EV sol
    vars2 = add_2nd_stage_variables(mdl, pars, intvars=intvars)

    # get numerical 1st stage and symbolical 2nd stage objectives
    mdl.setObjective(get_1st_stage_objective(pars, EV_vars1) +
                     get_2nd_stage_objective(pars, vars2), GRB.MINIMIZE)

    # add only 2nd stage constraints, as 1st were already handled by the EV.
    # The "demands" variable will be de facto used as a parameter of the
    # optimization, as it will be fixed with lb=ub
    demands = add_2nd_stage_constraints(mdl, pars, EV_vars1, vars2)

    # solve each scenario
    results = []
    S = samples.shape[0]  # number of scenarios
    for i in tqdm(range(S), total=S, desc='solving EEV'):
        # set demands to the i-th sample
        fix_var(mdl, demands, samples[i])

        # run optimization and save its result
        mdl.optimize()
        if mdl.Status != GRB.INFEASIBLE:
            results.append(mdl.ObjVal)

    mdl.dispose()
    return results


def optimize_TS(pars: Dict[str, np.ndarray],
                samples: np.ndarray,
                intvars: bool = False,
                verbose: int = 0) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    Computes the approximated Two-stage Recourse Model, where the continuous 
    distribution is discretazed via sampling.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    samples : np.ndarray
        Samples approximating the continuous distribution to a discrete one.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.

    Returns
    -------
    obj : float
        Value of the objective function at the optimal point.
    solution : dict[str, np.ndarray]
        Dictionary containing the value of 1st stage variable at the optimum.
    purchase_prob : float
        The probability, according to the TS solution and the sample, that a 
        purchase from an external source must be done.
    '''
    # get the number of scenarios
    S = samples.shape[0]

    # create large scale deterministic equivalent problem
    mdl = gb.Model(name='LSDE')
    if verbose < 1:
        mdl.Params.LogToConsole = 0

    # create 1st stage variables and 2nd stage variables, one per scenario
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)
    vars2 = add_2nd_stage_variables(mdl, pars, scenarios=S, intvars=intvars)

    # set objective
    obj1 = get_1st_stage_objective(pars, vars1)
    obj2 = get_2nd_stage_objective(pars, vars2)
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    # set constraints
    add_1st_stage_constraints(mdl, pars, vars1)
    add_2nd_stage_constraints(mdl, pars, vars1, vars2, samples)

    # solve
    mdl.optimize()

    # return the solution
    objval = mdl.ObjVal
    convert = ((lambda o: util.var2val(o).astype(int))
               if intvars else
               (lambda o: util.var2val(o)))
    sol1 = {name: convert(var) for name, var in vars1.items()}
    sol2 = {name: convert(var) for name, var in vars2.items()}

    # compute the purchase probability
    purchase_prob = (sol2['Y+'] > 0).sum() / sol2['Y+'].size

    # return
    mdl.dispose()
    return objval, sol1, purchase_prob


def run_MRP(pars: Dict[str, np.ndarray],
            solution: Dict[str, np.ndarray],
            sample_size: int,
            alpha: float = 0.95,
            replicas: int = 30,
            intvars: bool = False,
            verbose: int = 0,
            seed: int = None) -> float:
    '''
    Applies the MRP to a given solution to compute its confidence interval.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    solution : dict[str, np.ndarry]
        Dictionary containing a solution to the problem.
    sample_size : int
        Size of the samples to draw.
    alpha : flaot, optiona
        Confidence percentage for the MRP.
    replicas : int
        Number of replications for the MRP.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.
    seed : int
        Random seed for LHS.

    Returns
    -------
    CI : float
        An upper bound on the optimality gap of the given solution.
    '''
    # using the MRP basically mean computing N times the LSTDE
    # create large scale deterministic equivalent problem
    n, s = pars['n'], pars['s']
    S = sample_size
    mdl = gb.Model(name='LSDE')
    if verbose < 2:
        mdl.Params.LogToConsole = 0

    # create 1st stage variables and 2nd stage variables, one per scenario
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)
    vars2 = add_2nd_stage_variables(mdl, pars, scenarios=S, intvars=intvars)

    # set objective
    obj1 = get_1st_stage_objective(pars, vars1)
    obj2 = get_2nd_stage_objective(pars, vars2)
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    # set constraints
    add_1st_stage_constraints(mdl, pars, vars1)
    demands = add_2nd_stage_constraints(mdl, pars, vars1, vars2)

    # create also a submodel to solve only the second stage problem
    sub = gb.Model(name='sub')
    if verbose < 2:
        sub.Params.LogToConsole = 0

    # add 2nd stage variables and only X from 1st stage
    sub_X = np.array(
        sub.addVars(n, s, lb=0, ub=0, name='X').values()).reshape(n, s)
    sub_vars2 = add_2nd_stage_variables(sub, pars, intvars=intvars)

    # set objective, only 2nd stage
    sub.setObjective(get_2nd_stage_objective(pars, sub_vars2), GRB.MINIMIZE)

    # set constraints, only 2nd stage
    sub_demand = add_2nd_stage_constraints(sub, pars, {'X': sub_X}, sub_vars2)

    # start the MRP
    G = []
    for r in tqdm(range(replicas), total=replicas, desc='MRP iteration'):
        # draw a sample - cannot pass the same seed, otherwise the same samples
        # will be drawn again and the CI will be zero
        sample = util.draw_samples(S, pars, asint=intvars, seed=seed * (r + 1))

        # fix the problem's demands to this sample
        fix_var(mdl, demands, sample)

        # solve the problem
        mdl.optimize()
        vars1_k = {name: util.var2val(var) for name, var in vars1.items()}

        # calculate G_k
        G_k = 0
        for i in tqdm(range(S), total=S, desc='Computing G  ', leave=False):
            # solve v(w_k_s, x_hat)
            fix_var(sub, sub_demand, sample[i])
            fix_var(sub, sub_X, solution['X'])
            sub.optimize()
            a = get_1st_stage_objective(pars, solution).getValue() + sub.ObjVal

            # solve v(w_k_s, x_k)
            fix_var(sub, sub_X, vars1_k['X'])
            b = get_1st_stage_objective(pars, vars1_k).getValue() + sub.ObjVal

            # accumulate in G_k
            G_k += (a - b) / S

        G.append(G_k)

    mdl.dispose()
    sub.dispose()

    # compute the confidence interval
    G = np.array(G)
    G_bar, sG = G.mean(), G.std(ddof=1)
    eps = stats.t.ppf(alpha, replicas - 1) * sG / np.sqrt(replicas)
    return G_bar + eps


def optimize_WS(pars: Dict[str, np.ndarray],
                samples: np.ndarray,
                intvars: bool = False,
                verbose: int = 0) -> List[float]:
    '''
    Computes the Wait-and-See solution.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    samples : np.ndarray
        Samples approximating the continuous distribution to a discrete one.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.

    Returns
    -------
    objs : list[float]
        A list with all the objectives for each sample.
    '''
    # create the wait-and-see model
    mdl = gb.Model(name='LSDE')
    if verbose < 1:
        mdl.Params.LogToConsole = 0

    # create 1st and 2nd stage variables
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)
    vars2 = add_2nd_stage_variables(mdl, pars, intvars=intvars)

    # set objective (now we optimize over vars1 as well, whereas in EEV we
    # used the EV solution)
    mdl.setObjective(get_1st_stage_objective(pars, vars1) +
                     get_2nd_stage_objective(pars, vars2), GRB.MINIMIZE)

    # set constraints
    add_1st_stage_constraints(mdl, pars, vars1)
    demands = add_2nd_stage_constraints(mdl, pars, vars1, vars2)

    # solve each scenario
    results = []
    S = samples.shape[0]  # number of scenarios
    for i in tqdm(range(S), total=S, desc='solving WS'):
        # set demands to the i-th sample
        fix_var(mdl, demands, samples[i])

        # run optimization and save its result
        mdl.optimize()
        if mdl.Status != GRB.INFEASIBLE:
            results.append(mdl.ObjVal)
    return results


def labor_sensitivity_analysis(
        pars: Dict[str, np.ndarray],
        samples: np.ndarray,
        factors: List[float],
        intvars: bool = False,
        verbose: int = 0) -> Tuple[Dict[Tuple[float, float], float], np.ndarray]:
    '''
    Computes the sensitivity analysis of the TS solution with respect to 
    variations to the labour extra capacity increase upper bound and cost. 
    These two parameters are changed by some factor, and the new TS solution 
    computed.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    samples : np.ndarray
        Samples approximating the continuous distribution to a discrete one.
    factors : list[float]
        A list of factors for which the TS solution sensitivity is computed.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.

    Returns
    -------
    results : dict[tuple[float, float], float]
        A dictionary containing for each combination of two modification 
        factors the corresponding TS objective value.
    grid : np.ndarray
        The same data, but arranged in a grid.
    '''

    # get all the combinations of factors
    F = np.array(factors).flatten()
    L = F.size

    # instead of having UB and C2 as constants, we create two variables and fix
    # them to different values for each factor combination
    UBs = np.tile(pars['UB'], (L, 1, 1)).astype(float)
    C2s = np.tile(pars['C2'], (L, 1, 1)).astype(float)

    # modify each UB and C2 on their corresponding couple of factors. Modify
    # only labor, i.e., last row
    UBs[:, -1] *= F[:, None]
    C2s[:, -1] *= F[:, None]

    ####################### CODE ALMOST IDENTICAL TO TS #######################
    # get the number of scenarios
    S = samples.shape[0]
    m, s = pars['m'], pars['s']

    # create large scale deterministic equivalent problem
    mdl = gb.Model(name='LSDE')
    if verbose < 1:
        mdl.Params.LogToConsole = 0

    # now, make a copy of the original paramters and replace the original UB
    # and C2 with some variables
    pars = deepcopy(pars)
    pars['UB'] = np.array(mdl.addVars(m, s, name='UB').values()).reshape(m, s)
    pars['C2'] = np.array(mdl.addVars(m, s, name='C2').values()).reshape(m, s)

    # create 1st stage variables and 2nd stage variables, one per scenario
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)
    vars2 = add_2nd_stage_variables(mdl, pars, scenarios=S, intvars=intvars)

    # set objective
    obj1 = get_1st_stage_objective(pars, vars1)
    obj2 = get_2nd_stage_objective(pars, vars2)
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    # set constraints
    add_1st_stage_constraints(mdl, pars, vars1)
    add_2nd_stage_constraints(mdl, pars, vars1, vars2, samples)

    # solve TS for each modification combination
    results = {}
    for i, j in tqdm(product(range(L), range(L)),
                     total=L**2, desc='labor sensitivity'):
        # fix UB and C2 to their modified counterpart
        fix_var(mdl, pars['UB'], UBs[i])
        fix_var(mdl, pars['C2'], C2s[j])

        # solve and save
        mdl.optimize()
        results[(F[i], F[j])] = mdl.ObjVal
    return results, np.array(list(results.values())).reshape(L, L)

    ################ MULTIPROCESSING CODE - NOT REALLY FASTER #################
    # from multiprocessing import Pool
    # from copy import deepcopy
    # from functools import partial

    # # get all the combinations
    # L = len(factors)
    # F1, F2 = np.meshgrid(factors, factors)

    # # for each combinations, modify the extra capacity upper bound and cost
    # pars_modified = []
    # for i, j in itertools.product(range(L), range(L)):
    #     pars_ = deepcopy(pars)
    #     pars_['UB'][-1] = pars_['UB'][-1] * F1[i, j]  # last row is labour
    #     pars_['C2'][-1] = pars_['C2'][-1] * F2[i, j]  # last row is labour
    #     pars_modified.append(pars_)

    # # compute the TS solution for each new set of parameters in parallel
    # f = partial(optimize_TS, samples=samples, intvars=intvars,
    #             verbose=verbose - 1)
    # results = []
    # p = Pool()
    # with p:
    #     for r in tqdm(p.imap(f, pars_modified), total=L**2,
    #                   desc='sensivitiy analysis'):
    #         results.append(r[0])
    # return np.array(results).reshape(L, L)


def dep_sensitivity_analysis(
        pars: Dict[str, np.ndarray],
        samplesize: int,
        factors: List[float],
        intvars: bool = False,
        verbose: int = 0,
        seed: int = 0) -> Tuple[Dict[Tuple[float, float], float], np.ndarray]:
    '''
    Computes the sensitivity analysis of the TS solution with respect to 
    the inter-product and inter-period dependence. These two correlations are 
    changed by some factor, and the new TS solution computed.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    samplesize : int
        Size of the samples for approximating the continuous distribution to a 
        discrete one.
    factors : list[float]
        A list of factors for which the TS solution sensitivity is computed.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : int, optional
        Verbosity level of Gurobi model. Defaults to 0, i.e., no verbosity.
    seed : int, optional
        Random number generator seed.

    Returns
    -------
    results : dict[tuple[float, float], float]
        A dictionary containing for each combination of two correlation factors 
        the corresponding TS objective value.
    grid : np.ndarray
        The same data, but arranged in a grid.
    '''

    F = np.array(factors).flatten()
    L = F.size

    # grab some parameters
    S = samplesize
    n, s = pars['n'], pars['s']
    meanM = pars['demand_mean'].flatten()
    stds = pars['demand_std']

    # build the multivariate normal covariance matrix for each combination of
    # the factors
    covarM = np.empty((L, L, n, s, n, s))
    for f1, f2, i1, t1, i2, t2 in product(*[range(d) for d in covarM.shape]):
        # self covariance
        if i1 == i2 and t1 == t2:
            factor = 1
        # inter-product correlation in the same period
        elif t1 == t2:
            factor = F[f1]
        # inter-period correlation for the same product
        elif i1 == i2:
            factor = F[f2]
        # no correlation between different products at different periods
        else:
            factor = 0

        # asign to matrix
        covar = stds[i1, t1] * stds[i2, t2] * factor**2 * np.sign(factor)
        covarM[f1, f2, i1, t1, i2, t2] = covar

    # reshape into proper form
    covarM = covarM.reshape(L, L, n * s, n * s)
    assert all((np.linalg.eig(covarM[i, j])[0] >= 0).all()
               for i in range(L) for j in range(L))

    # draw all the necessary samples
    rng = np.random.default_rng(seed=seed)
    samples = np.empty((L, L, S, n, s))
    for i, j in product(range(L), range(L)):
        sample = rng.multivariate_normal(meanM, covarM[i, j], size=S)
        samples[i, j] = sample.reshape(S, n, s)
    if intvars:
        samples = samples.astype(int)

    # create large scale deterministic equivalent problem
    mdl = gb.Model(name='LSDE')
    if verbose < 1:
        mdl.Params.LogToConsole = 0

    # create 1st stage variables and 2nd stage variables, one per scenario
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)
    vars2 = add_2nd_stage_variables(mdl, pars, scenarios=S, intvars=intvars)

    # set objective
    obj1 = get_1st_stage_objective(pars, vars1)
    obj2 = get_2nd_stage_objective(pars, vars2)
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    # set constraints
    add_1st_stage_constraints(mdl, pars, vars1)
    demands = add_2nd_stage_constraints(mdl, pars, vars1, vars2)

    # solve TS for each combination of crosscorrelation
    results = {}
    for i, j in tqdm(product(range(L), range(L)),
                     total=L**2, desc='demand sensitivity'):
        # fix the demands to the current sample
        fix_var(mdl, demands, samples[i, j])

        # solve and save (averaging over the number of replications)
        mdl.optimize()
        results[(F[i], F[j])] = mdl.ObjVal
    return results, np.array(list(results.values())).reshape(L, L)
