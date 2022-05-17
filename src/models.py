import numpy as np
import gurobipy as gb
from gurobipy import GRB
import itertools
from typing import Union, Dict, Tuple


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
    stage2: bool = True
) -> Tuple[Union[gb.LinExpr, float], Union[gb.LinExpr, float]]:
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

        # pick variables for this stage
        if stage == 0:  # i.e., 1st stage
            vars_ = [vars['X'], vars['U'], vars['Z+'], vars['Z-']]
            costs_ = [pars['C1'], pars['C2'], pars['C3+'], pars['C3-']]
        else:
            vars_ = [vars['Y+'], vars['Y-']]
            costs_ = [pars['Q+'], pars['Q-']]

        # compute the actual cost
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
                ranges = range(cost.shape[0]), range(cost.shape[1])
                objs[stage] += gb.quicksum(
                    cost[i, j] * var[i, j]
                    for i, j in itertools.product(*ranges))
    return tuple(objs)


def add_first_stage_constraints(mdl: gb.Model,
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
    for k, (j, t) in enumerate(itertools.product(range(m), range(s))):
        AX = gb.quicksum(A[i, j] * X[i, t] for i in range(n))
        mdl.addLConstr(AX <= B[j, t] + U[j, t], name=f'con_production_{k}')

    # work force level increase/decrease
    for k, t in enumerate(range(1, s)):
        AXX = gb.quicksum(A[i, -1] * (X[i, t] - X[i, t - 1]) for i in range(n))
        mdl.addLConstr(Zp[t - 1] - Zm[t - 1] == AXX, name=f'con_workforce_{k}')

    # extra capacity upper bounds
    UB = pars['UB']
    for k, (j, t) in enumerate(itertools.product(range(m), range(s))):
        mdl.addLConstr(U[j, t] <= UB[j, t], name=f'con_extracap_{k}')

    mdl.update()
    return


def add_second_stage_expval_constraints(mdl: gb.Model,
                                        pars: Dict[str, np.ndarray],
                                        vars: Dict[str, gb.MVar]) -> None:
    '''
    Adds 2nd stage constraints with expected values in place of random 
    variables to the model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model to add constraints to.
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, gurobipy.MVar]
        Dictionary containing the optimization variables.
    '''
    s, n = pars['s'], pars['n']
    X, Yp, Ym = vars['X'], vars['Y+'], vars['Y-']
    d_mean = pars['demand_mean']
    for k, (i, t) in enumerate(itertools.product(range(n), range(s))):
        mdl.addLConstr(
            X[i, t] + Ym[i, t - 1] + Yp[i, t] - Ym[i, t] == d_mean[i, t],
            name=f'con_demand_{k}')


def optimize_EV_solution(
        pars: Dict[str, np.ndarray],
        intvars: bool = False,
        verbose: bool = False) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    Computes the Expected Value solution via a Gurobi model.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.
    verbose : bool, optional
        Switch for verbosity of Gurobi model. Defaults to False.

    Returns
    -------
    obj : float
        Value of the objective function at the optimal point.
    solution : dict[str, np.ndarray]
        Dictionary containing the value of each variable at the optimum.
    '''
    # initialize model
    mdl = gb.Model(name='EV')
    if not verbose:
        mdl.Params.LogToConsole = 0

    # create the variables
    vars_ = add_variables(mdl, pars, intvars=intvars)

    # get the 1st and 2nd stage objectives, and sum them up (avoiding Nones)
    objs = get_objective(pars, vars_)
    mdl.setObjective(gb.quicksum(objs), GRB.MINIMIZE)

    # add 1st stage constraints
    add_first_stage_constraints(mdl, pars, vars_)

    # to get the EV Solution, in the second stage constraints the random
    # variables are replaced with their expected value
    add_second_stage_expval_constraints(mdl, pars, vars_)

    # run optimization
    # mdl.update()
    mdl.optimize()

    # retrieve optimal objective value and optimal variables
    optimal_obj = mdl.getAttr(GRB.Attr.ObjVal)
    solution = {
        name: var.getAttr(GRB.Attr.X) for name, var in vars_.items()
    }
    return optimal_obj, solution
