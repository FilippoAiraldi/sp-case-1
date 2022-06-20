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

#set parameters and decision variables
s=4 #periods
n=2 #products
m=3 #production sources

demand_mean= np.array([[30, 32, 44, 48],
                       [50, 50, 50, 60]])*10
demand_std=np.tile(np.array([[45], [75]]), (1, s)) #demand standard deviation

#print(np.asarray(demand_mean).flatten().size) convert to array and size for sampling
A=np.array([[4, 3, 3],
            [5, 2, 7]])  # unit source j needed for product i

#upper bound on extra capacity of source j at time t
UB=np.array([[40, 40, 40, 35],
                        [30, 30, 25, 35],
                        [45, 45, 37.5, 35]]) * 10 

C1 = np.tile(np.array([[100], [150]]), (1, 4))
C2 = np.tile(np.array([[15], [20], [10]]), (1, 4))
C3 = np.array([20,20,20]) #Increase work force level
C4 = np.array([15,15,15])   # decrease worke force level
Q_p= np.tile(np.array([[400], [450]]), (1, 4)) # unit cost for shortage
Q_m= np.matrix('25 25 25 100; 30 30 30 150')  #unit cost for suprlus

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
alpha=0.5 # we use 0.5,0.6,0.9,1.0
lp1=gb.Model('interior point') # Convex Lp in (11) that finds interior point Y^{-}

#decison variables (same as for the recourse)
vars1=add_1st_stage_variables(lp1, pars, intvars=intvars)
vars2 = add_2nd_stage_variables(lp1, pars, intvars=intvars)


#objective function \alpha
lp1.setObjective(alpha,GRB.MAXIMIZE)

#Constraints
add_1st_stage_constraints(lp1, pars, vars1) # deterministic constraint

#Constraint T\Lambda -Y^-\geq 0 is by using second stage constarints modify

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
    con = (X + Yp  - demands).flatten()    #modified
    mdl.addConstrs((con[i] >= 0 for i in range(con.size)), name='con_demand')
    if return_:
        return demands
 #Constraint Y^{-}+M_i \delta \geq omega
 


#run model
lp1.optimize()

#save values and use the solution Y^{-} for the line search



#Function to add feasible cuts ----TODO



#set Convergence criteria and start the while loop for algorithm ---TODO






#Function to add feasible cuts ---TODO

























