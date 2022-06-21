# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:08:39 2022

@author: Alban
"""

import gurobipy as gb
from gurobipy import GRB
import math
import numpy as np
from scipy.optimize import line_search
from scipy.stats import norm
from scipy.stats.distributions import truncnorm
from scipy.optimize import minimize
from scipy.linalg import block_diag
from math import *
from scipy.stats import qmc
from pyDOE import *
import matplotlib.pyplot as plt
import util
from copy import deepcopy
from itertools import product
from typing import Union, Dict, Tuple, List, Optional
#set the model
cc=gb.Model('ChanceConstraints')


#Get parameters from util
pars=util.get_parameters()

N_blo=np.array([[1,0,0,0],
            [1,1,0,0],
            [1,1,1,0],
            [1,1,1,1]])
#Transformation matrix for random variable \xi to \omega
N_block=np.kron(np.eye(2,dtype=int),N_blo)    



"""#Sampling and computing the CDF for the From UTIL
K = 20 #number of scenarios
demand_mean_reshape=demand_mean.reshape(-1,8)    #.tolist()
demand_std_reshape=demand_std.reshape(-1,8)    #in case of list.tolist() 
sampler=lhs(8,samples=K) #latin hypercube sampling 8 is the array length of demand_mean(converted to array)

for i in range(8):
    sampler[:, i]=norm(loc=demand_mean_reshape[i], scale=demand_std_reshape[i]).ppf(sampler[:, i])
sampler
"""

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
                              ) -> Optional[np.ndarray]:
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
    demands : np.ndarray of gurobipy.Var
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
    con = (X + Yp  - demands).flatten()
    mdl.addConstrs((con[i] >= 0 for i in range(con.size)), name='con_demand')

    if return_:
        return demands

'''
def add_constraint(mdl: gb.Model,
                              pars: Dict[str, np.ndarray],
                              vars_1st: Dict[str, np.ndarray],
                              vars_2nd: Dict[str, np.ndarray],
                              demands: np.ndarray = None
                              ) -> Optional[np.ndarray]:
  
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
    demands : np.ndarray of gurobipy.Var
        The new demand variables used in the constraints. Only created and 
        returned when no demand is passed in the arguments.

    # get the number of scenarios (if 2D, then 1; if 3D, then first dimension)
    S = 1 if vars_2nd['Y+'].ndim == 2 else vars_2nd['Y+'].shape[0]

    s, n = pars['s'], pars['n']
    Mi=1000 
    X, Yp, Ym = vars_1st['X'], vars_2nd['Y+'], vars_2nd['Y-']
    
    #Transform demands and Ym
    Ym=np.matmul(N_block,Ym) #transform YM
    demand=np.matmul(N_block,demand) #Transform 

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
    delta=np.where(Ym-demands>=0,0,1) #indicator function
    con = (Ym[...,1]  +Mi*delta[]- demands[:,0]).flatten()
    mdl.addConstrs((con[i] >= 0 for i in range(con.size)), name='con_demand')

    if return_:
        return demands
''' 

#function to compute the Cumulative normal distribution
def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

#Function to compute the 1D pdf for normal distribution with mean an standard deviaition sd
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


#function to compute partial derivatives of F 


#def partialNormCDF(z,):
    

#function to compute the solution for the LP  (11) in report
alph=0.5 # we use 0.5,0.6,0.9,1.0
lp1=gb.Model('interior point') # Convex Lp in (11) that finds interior point Y^{-}

#decison variables (same as for the recourse)
vars1=add_1st_stage_variables(lp1, pars, intvars=False)
vars2=add_2nd_stage_variables(lp1, pars)


#objective function \alpha
lp1.setObjective(alph,GRB.MAXIMIZE)

#Constraints

 # deterministic constraint
add_1st_stage_constraints(lp1, pars, vars1)

#Constraint T\Lambda -Y^-\geq 0 is by using modifyed second stage constraints 
add_2nd_stage_constraints(lp1, pars, vars1, vars2)  


#Constraint Y^{-}+M_i \delta \geq omega
 
#add_constraint(lp1,pars,vars1,vars2)


#Constraint \sum_{i}p_i\delta^{j}\leq 1-alph



#run model
lp1.optimize()

#save values and use the solution Y^{-} for the line search
objval = lp1.ObjVal
convert = ((lambda o: util.var2val(o)))
sol1 = {name: convert(var) for name, var in vars1.items()}
sol2 = {name: convert(var) for name, var in vars2.items()}
sol2_avg = {f'{name} avg': var.mean(0) for name, var in sol2.items()}





#line search using sol2['Y-']











#Function to add feasible cuts TODO

























