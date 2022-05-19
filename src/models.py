import numpy as np
import gurobipy as gb
from gurobipy import GRB
from itertools import product
from tqdm import tqdm
from typing import Union, Dict, Tuple, List, Optional


def add_1st_stage_variables(mdl: gb.Model,
                            pars: Dict[str, np.ndarray],
                            intvars: bool = False) -> Dict[str, gb.MVar]:
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
    vars : dict[str, gurobipy.MVar]
        Dictionary containing the 1st stage variables as matrices.
    '''
    TYPE = GRB.INTEGER if intvars else GRB.CONTINUOUS
    s, n, m = pars['s'], pars['n'], pars['m']

    vars_ = [('X', (n, s)), ('U', (m, s)), ('Z+', s - 1), ('Z-', s - 1)]
    return {
        name: mdl.addMVar(size, lb=0, vtype=TYPE, name=name)
        for name, size in vars_
    }


def add_2nd_stage_variables(mdl: gb.Model,
                            pars: Dict[str, np.ndarray],
                            intvars: bool = False) -> Dict[str, gb.MVar]:
    '''
    Creates the 2nd stage variables for the current Gurobi model.

    Parameters
    ----------
    mdl : gurobipy.Model
        Model for which to create the 2nd stage variables.
    pars : dict[str, int | np.ndarray]
        Dictionary containing the optimization problem parameters.
    intvars : bool, optional
        Some of the variables are constrained to integers. Otherwise, they are 
        continuous.

    Returns
    -------
    vars : dict[str, gurobipy.MVar]
        Dictionary containing the 2nd stage variables as matrices.
    '''
    TYPE = GRB.INTEGER if intvars else GRB.CONTINUOUS
    s, n, m = pars['s'], pars['n'], pars['m']

    vars_ = [('Y+', (n, s)), ('Y-', (n, s))]
    return {
        name: mdl.addMVar(size, lb=0, vtype=TYPE, name=name)
        for name, size in vars_
    }


def fix_var(var: gb.MVar, value: Union[float, np.ndarray]) -> None:
    '''
    Fixes a variable to a given value.

    Parameters
    ----------
    var : gurobipy.MVar
        The variable to be fixed.
    value : float, np.ndarray
        The value at which to fix the variable.
    '''
    var.setAttr(GRB.Attr.LB, value)
    var.setAttr(GRB.Attr.UB, value)


def get_1st_stage_objective(pars: Dict[str, np.ndarray],
                            vars: Dict[str, Union[gb.MVar, np.ndarray]]
                            ) -> Tuple[Union[gb.LinExpr, float], ...]:
    '''
    Computes the 1st stage objective.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, gurobipy.MVar | np.ndarray]
        1st stage variables to use to compute the objective. Can be either 
        symbolical, or numerical.

    Returns
    -------
    obj :  gurobipy.LinExpr | float
        An expression (if variables are symbolical) or a number (if vars are
        umerical) representing the 1st stage objective.
    '''
    # NOTE: MVars do not support full matrix operations as numpy arrays. So, to
    # be safe, use for-loops and indices.

    obj = 0
    vars_ = [vars['X'], vars['U'], vars['Z+'], vars['Z-']]
    costs = [pars['C1'], pars['C2'], pars['C3+'], pars['C3-']]

    # compute the actual objective by multiplying variables by costs
    for var, cost in zip(vars_, costs):
        # if it is numpy, use matrix element-wise op
        if isinstance(var, np.ndarray):
            obj += float((cost * var).sum())

        # if it is symbolical, use loop
        elif cost.ndim == 1:
            obj += gb.quicksum(cost[i] * var[i]
                               for i in range(cost.shape[0]))
        else:
            obj += gb.quicksum(
                cost[i, j] * var[i, j]
                for i, j in product(
                    range(cost.shape[0]), range(cost.shape[1])))
    return obj


def get_2nd_stage_objective(pars: Dict[str, np.ndarray],
                            vars: Dict[str, Union[gb.MVar, np.ndarray]]
                            ) -> Tuple[Union[gb.LinExpr, float], ...]:
    '''
    Computes the 2nd stage objective.

    Parameters
    ----------
    pars : dict[str, np.ndarray]
        Dictionary containing the optimization problem parameters.
    vars : dict[str, gurobipy.MVar | np.ndarray]
        2nd stage variables to use to compute the objective. Can be either 
        symbolical, or numerical.

    Returns
    -------
    obj :  gurobipy.LinExpr | float
        An expression (if variables are symbolical) or a number (if vars are
        umerical) representing the 2nd stage objective.
    '''
    # NOTE: MVars do not support full matrix operations as numpy arrays. So, to
    # be safe, use for-loops and indices.

    obj = 0
    vars_ = [vars['Y+'], vars['Y-']]
    costs = [pars['Q+'], pars['Q-']]

    # compute the actual objective by multiplying variables by costs
    for var, cost in zip(vars_, costs):
        # if it is numpy, use matrix element-wise op
        if isinstance(var, np.ndarray):
            obj += float((cost * var).sum())

        # if it is symbolical, use loop
        elif cost.ndim == 1:
            obj += gb.quicksum(cost[i] * var[i]
                               for i in range(cost.shape[0]))
        else:
            obj += gb.quicksum(
                cost[i, j] * var[i, j]
                for i, j in product(
                    range(cost.shape[0]), range(cost.shape[1])))
    return obj


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
                              vars_1st: Dict[str, gb.MVar],
                              vars_2nd: Dict[str, gb.MVar],
                              demands: Union[np.ndarray, gb.MVar] = None
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
    vars_1st : dict[str, gurobipy.MVar]
        Dictionary containing the 1st stage optimization variables.
    vars_2nd : dict[str, gurobipy.MVar]
        Dictionary containing the 2nd stage optimization variables.
    demands : np.ndarray, optional
        Deterministic demand values. If None, a new variable is added to the 
        model.

    Returns
    -------
    demands : gurobipy.MVar
        The new demand variables used in the constraints. Only created and 
        returned when no demand is passed in the arguments.
    '''
    s, n = pars['s'], pars['n']
    X, Yp, Ym = vars_1st['X'], vars_2nd['Y+'], vars_2nd['Y-']

    if demands is None:
        return_ = True
        demands = mdl.addMVar((n, s), lb=0, ub=0, name='demand')
    else:
        return_ = False

    # in the first period, zero surplus is assumed
    for i, t in product(range(n), range(s)):
        Ym_previous = 0 if t == 1 else Ym[i, t - 1]
        mdl.addLConstr(
            X[i, t] + Ym_previous + Yp[i, t] - Ym[i, t] == demands[i, t],
            name=f'con_demand_{i}_{t}')
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
    convert = (lambda o: o.astype(int)) if intvars else (lambda o: o)
    sol1 = {name: convert(var.X) for name, var in vars1.items()}
    sol2 = {name: convert(var.X) for name, var in vars2.items()}
    return mdl.ObjVal, sol1, sol2


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
        fix_var(demands, samples[i])

        # run optimization and save its result
        mdl.optimize()
        if mdl.Status != GRB.INFEASIBLE:
            results.append(mdl.ObjVal)
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
    '''
    # get the number of scenarios
    S = samples.shape[0]

    # create large scale deterministic equivalent problem
    mdl = gb.Model(name='LSDE')
    if verbose < 1:
        mdl.Params.LogToConsole = 0

    # create 1st stage variables
    vars1 = add_1st_stage_variables(mdl, pars, intvars=intvars)

    # create one set of 2nd stage variables per scenario
    vars2 = [
        add_2nd_stage_variables(mdl, pars, intvars=intvars) for _ in range(S)
    ]

    # set objective
    obj1 = get_1st_stage_objective(pars, vars1)
    objs2 = [get_2nd_stage_objective(pars, var2) for var2 in vars2]
    mdl.setObjective(obj1 + (1 / S) * gb.quicksum(objs2), GRB.MINIMIZE)

    # set constraints
    add_1st_stage_constraints(mdl, pars, vars1)
    for s in range(S):
        add_2nd_stage_constraints(mdl, pars, vars1, vars2[s], samples[s])

    # solve
    mdl.optimize()

    # return the solution
    convert = (lambda o: o.astype(int)) if intvars else (lambda o: o)
    sol1 = {name: convert(var.X) for name, var in vars1.items()}
    return mdl.ObjVal, sol1


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
        fix_var(demands, samples[i])

        # run optimization and save its result
        mdl.optimize()
        if mdl.Status != GRB.INFEASIBLE:
            results.append(mdl.ObjVal)
    return results
