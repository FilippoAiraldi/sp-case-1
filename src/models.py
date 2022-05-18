import numpy as np
import gurobipy as gb
from gurobipy import GRB
from itertools import product
from tqdm import tqdm
from typing import Union, Dict, Tuple, List


def add_variables(mdl: gb.Model,
                  pars: Dict[str, np.ndarray],
                  stage1: bool = True,
                  stage2: bool = True,
                  intvars: bool = False) -> Dict[str, gb.MVar]:
    '''
    Creates the variables for the current Gurobi model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model for which to create the 1st stage variables.
    pars : dict[str, int | np.ndarray]
        Dictionary containing the optimization problem parameters.
    stage1 : bool, optional
        Whether to create the first stage variables or not. Defaults to true.
    stage2 : bool, optional
        Whether to create the second stage variables or not. Defaults to true.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.

    Returns
    -------
    vars : dict[str, gurobipy.MVar]
        Dictionary containing the 1st and/or 2nd stage variables as matrices.
    '''
    # decide if continuous or integers
    TYPE = GRB.INTEGER if intvars else GRB.CONTINUOUS

    # get some parameters
    s, n, m = pars['s'], pars['n'], pars['m']

    # build 1st and 2nd stage variable names, size and type
    vars_ = []
    if stage1:
        vars_ += [('X', (n, s)), ('U', (m, s)), ('Z+', s - 1), ('Z-', s - 1)]
    if stage2:
        vars_ += [('Y+', (n, s)), ('Y-', (n, s))]

    # instantiate variables and put them in a dictionary
    return {
        name: mdl.addMVar(size, lb=0, vtype=TYPE, name=name)
        for name, size in vars_
    }


def get_objective(
        pars: Dict[str, np.ndarray],
        vars: Dict[str, Union[gb.MVar, np.ndarray]],
        stage1: bool = True,
        stage2: bool = True) -> Tuple[Union[gb.LinExpr, float], ...]:
    '''
    Computes the objective.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, gurobipy.MVar | np.ndarray]
        Variables to use to compute the objective. Can be either symbolical,
        or numerical.
    stage1 : bool, optional
        Whether to include first stage costs. Defaults to true.
    stage2 : bool, optional
        Whether to include second stage costs. Defaults to true.

    Returns
    -------
    obj_stage1 :  gurobipy.LinExpr | float
        An expression (if variables are symbolical) or a number (if vars are
        umerical) representing the objective of 1st stage variables.
    obj_stage2 :  gurobipy.LinExpr | float
        An expression (if variables are symbolical) or a number (if vars are
        umerical) representing the objective of 2nd stage variables.
    '''
    # NOTE: MVars do not support full matrix operations as numpy arrays. So, to
    # be safe, use for-loops and indices.

    objs = [0.0, 0.0]

    for stage, do in enumerate((stage1, stage2)):
        # if current stage is not requested, skip it
        if not do:
            continue

        # pick variables and costs for this stage
        if stage == 0:  # i.e., 1st stage
            vars_ = [vars['X'], vars['U'], vars['Z+'], vars['Z-']]
            costs_ = [pars['C1'], pars['C2'], pars['C3+'], pars['C3-']]
        else:
            vars_ = [vars['Y+'], vars['Y-']]
            costs_ = [pars['Q+'], pars['Q-']]

        # compute the actual objective by multiplying variables by costs
        objs[stage] = 0
        for var, cost in zip(vars_, costs_):
            # if it is numpy, use matrix element-wise op
            if isinstance(var, np.ndarray):
                objs[stage] += float((cost * var).sum())

            # if it is symbolical, use loop
            elif cost.ndim == 1:
                objs[stage] += gb.quicksum(cost[i] * var[i]
                                           for i in range(cost.shape[0]))
            else:
                objs[stage] += gb.quicksum(
                    cost[i, j] * var[i, j]
                    for i, j in product(
                        range(cost.shape[0]), range(cost.shape[1])))
    return tuple(objs)


def add_1st_stage_constraints(mdl: gb.Model,
                              pars: Dict[str, np.ndarray],
                              vars: Dict[str, gb.MVar]) -> None:
    '''
    Adds 1st stage constraints to the model.

    Parameters
    ----------
    mdl : gurobipy
        Model to add constraints to.
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, gurobipy.MVar]
        Dictionary containing the optimization variables.
    '''
    X, U, Zp, Zm = vars['X'], vars['U'], vars['Z+'], vars['Z-']
    s, n, m = pars['s'], pars['n'], pars['m']

    # sufficient sources to produce necessary products
    A, B = pars['A'], pars['B']
    for j, t in product(range(m), range(s)):
        AX = gb.quicksum(A[i, j] * X[i, t] for i in range(n))
        mdl.addLConstr(AX <= B[j, t] + U[j, t], name=f'con_production_{j}_{t}')

    # work force level increase/decrease
    for t in range(1, s):
        AXX = gb.quicksum(A[i, -1] * (X[i, t] - X[i, t - 1]) for i in range(n))
        mdl.addLConstr(Zp[t - 1] - Zm[t - 1] == AXX, name=f'con_workforce_{t}')

    # extra capacity upper bounds
    UB = pars['UB']
    for j, t in product(range(m), range(s)):
        mdl.addLConstr(U[j, t] <= UB[j, t], name=f'con_extracap_{j}_{t}')

    mdl.update()
    return


def add_2nd_stage_constraints(mdl: gb.Model,
                              pars: Dict[str, np.ndarray],
                              vars: Dict[str, gb.MVar],
                              demands: Union[np.ndarray, gb.MVar]) -> None:
    '''
    Adds 2nd stage constraints with some deterministic values in place of 
    random demand variables to the model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model to add constraints to.
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, gurobipy.MVar]
        Dictionary containing the optimization variables.
    demands : np.ndarray
        Deterministic demand values.
    '''
    s, n = pars['s'], pars['n']
    X, Yp, Ym = vars['X'], vars['Y+'], vars['Y-']

    # in the first period, zero surplus is assumed
    for i, t in product(range(n), range(s)):
        Ym_previous = 0 if t == 1 else Ym[i, t - 1]
        mdl.addLConstr(
            X[i, t] + Ym_previous + Yp[i, t] - Ym[i, t] == demands[i, t],
            name=f'con_demand_{i}_{t}')


def optimize_EV(pars: Dict[str, np.ndarray],
                intvars: bool = False,
                verbose: int = 0) -> Tuple[float, Dict[str, np.ndarray]]:
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
    vars_ = add_variables(mdl, pars, intvars=intvars)

    # get the 1st and 2nd stage objectives, and sum them up
    objs = get_objective(pars, vars_)
    mdl.setObjective(gb.quicksum(objs), GRB.MINIMIZE)

    # add 1st stage constraints
    add_1st_stage_constraints(mdl, pars, vars_)

    # to get the EV Solution, in the second stage constraints the random
    # variables are replaced with their expected value
    d_mean = pars['demand_mean']
    if intvars:
        d_mean = d_mean.astype(int)
    add_2nd_stage_constraints(mdl, pars, vars_, demands=d_mean)

    # run optimization
    # mdl.update()
    mdl.optimize()

    # retrieve optimal objective value and optimal variables
    convert = (lambda o: o.astype(int)) if intvars else (lambda o: o)
    optimal_obj = mdl.ObjVal
    solution = {
        name: convert(var.X) for name, var in vars_.items()
    }
    return optimal_obj, solution


def optimize_EEV(pars: Dict[str, np.ndarray],
                 EV_solution: Dict[str, np.ndarray],
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
    EV_solution : dict[str, np.ndarry]
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
    vars_ = EV_solution | add_variables(mdl, pars, stage1=False,
                                        intvars=intvars)

    # get numerical 1st stage and symbolical 2nd stage objectives
    objs = get_objective(pars, vars_)
    mdl.setObjective(gb.quicksum(objs), GRB.MINIMIZE)

    # add only 2nd stage constraints, as 1st were already handled by the EV sol
    # the "demands" variable will be de facto used as a parameter of the
    # optimization, as it will be fixed with lb=ub
    demands = mdl.addMVar(
        (pars['n'], pars['s']), lb=0.0, ub=0.0, name='demands')
    add_2nd_stage_constraints(mdl, pars, vars_, demands=demands)

    # grab the list of constraints, these will be updated in the loop
    # mdl.update() # necessary to update so as to grab the newest constraints
    # cons = mdl.getConstrs()
    X_EV = EV_solution['X']

    # solve each scenarion
    results = []
    scenarios = samples.shape[0]
    for i in tqdm(range(scenarios), total=scenarios, desc='solving EEV'):
        # mdl.reset()

        # set demands to i-th sample
        sample_i = samples[i]
        demands.setAttr(GRB.Attr.LB, sample_i)
        demands.setAttr(GRB.Attr.UB, sample_i)

        # run optimization and save its result
        # mdl.update()
        mdl.optimize()
        if mdl.Status != GRB.INFEASIBLE:
            results.append(mdl.ObjVal)
    return results


def optimize_TS(pars: Dict[str, np.ndarray],
                samples: np.ndarray,
                init_lb: float = 0,
                tol: float = 1e-1,
                max_iter: int = int(1e6),
                intvars: bool = False,
                verbose: int = 0) -> Tuple[float, Dict[str, np.ndarray]]:

    # create master problem
    master = gb.Model(name='Master')
    if verbose < 2:
        master.Params.LogToConsole = 0

    # create master's variables (only 1st stage, and epigraph theta)
    master_vars = add_variables(master, pars, stage2=False, intvars=intvars)
    theta = master.addVar(lb=init_lb, name='theta')

    # set master's objective
    obj1, _ = get_objective(pars, master_vars, stage2=False)
    master.setObjective(obj1 + theta, GRB.MINIMIZE)

    # set master's constraints (no subgradient constraint at start)
    add_1st_stage_constraints(master, pars, master_vars)

    # create (generic) subproblem

    # start the L-shape algorithm
    for _ in tqdm(range(max_iter), total=max_iter, desc='L-shape algorithm'):
        # solve master problem and grab variable values
        master.optimize()
        vars_t = {name: var.X for name, var in master_vars.items()}
        theta_t = theta.X

        # solve subproblem, i.e., Q(x_t)

    # convert variables to int if requested
    return None, None
