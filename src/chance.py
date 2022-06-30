# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:08:28 2022

@author: Alban
"""

import gurobipy as gb
from gurobipy import GRB
import numpy as np
import scipy
import util
import math
from math import *
from pyDOE import *
import recourse
from itertools import product
from typing import Union, Dict, Tuple, List, Optional
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import linprog
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
    #added variable tau for maximization
    tau=mdl.addVar(vtype=GRB.CONTINUOUS, name='tau')

    # create binary variables
    delta = np.array(mdl.addVars(N, vtype=GRB.BINARY, name='delta').values())

    # add each constraint in the order in which they are written
    Ym = vars2['Y-'].reshape(-1, 1)

    # add constraint 1
    cons = [(Ym[i] + M[i] * delta[j] - demands[j, i]).item()
            for j, i in product(range(N), range(p))]
    mdl.addConstrs((cons[i] >= 0 for i in range(len(cons))), name='con_1')

    # add constraint 2
    mdl.addConstr(1 / N * delta.sum() <= 1 - tau-alpha, name='con_2')

    # add constraint 3
    add_1st_stage_constraints(mdl, pars, vars1)
    # ... TODO ...

    # add constraint 4
    recourse.add_1st_stage_constraints(mdl, pars, vars1)

    # add constraint 5
    sigma_bar = np.sqrt(np.diag(demand_cov))
    con = (Ym - demand_mean + 3 * sigma_bar).flatten()
    mdl.addConstrs((con[i] >= 0 for i in range(con.size)), name='con_5')
    mdl.setObjective(tau,GRB.MAXIMIZE)

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
def run_LP_11_K(samples, pars):
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
    #added variable tau for maximization
    tau=mdl.addVar(vtype=GRB.CONTINUOUS, name='tau')

    # create binary variables
    delta = np.array(mdl.addVars(N, vtype=GRB.BINARY, name='delta').values())

    # add each constraint in the order in which they are written
    Ym = vars2['Y-'].reshape(-1, 1)

    # add constraint 1
    cons = [(Ym[i] + M[i] * delta[j] - demands[j, i]).item()
            for j, i in product(range(N), range(p))]
    mdl.addConstrs((cons[i] >= 0 for i in range(len(cons))), name='con_1')

    # add constraint 2
    mdl.addConstr(1 / N * delta.sum() <= 1 - tau-alpha, name='con_2')

    # add constraint 3
    add_1st_stage_constraints(mdl, pars, vars1)
    # ... TODO ...

    # add constraint 4
    recourse.add_1st_stage_constraints(mdl, pars, vars1)

    # add constraint 5
    sigma_bar = np.sqrt(np.diag(demand_cov))
    con = (Ym - demand_mean + 3 * sigma_bar).flatten()
    conu=(-Ym+demand_mean+3*sigma_bar).flatten()
    mdl.addConstrs((con[i] >= 0 for i in range(con.size)), name='con_5')
    
    mdl.setObjective(tau,GRB.MAXIMIZE)

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
#Function to compute the 1D pdf for normal distribution with mean an standard deviaition sd
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def multi_gradient(samples,Y):
    
    '''
    function that computes the gradient of \alpha -F(Y) 
    Parameters
    ----------
    Y : TYPE     numpy.array or pars['Y'] as explained in util
        DESCRIPTION. you insert an array Y of shape (1,8) 
    Returns
    Samples: from util
    -------
    None.
    
    '''
    demands, demand_mean, demand_cov = samples
    #partial_F=np.zeros((1,8))
    partial_F=[]
    #std=np.sqrt(np.diag(demand_cov))
    std=np.linalg.det(demand_cov) ** 8
    index=[0,1,2,3,4,5,6,7]
    for i, j in product(range(7),range(7)):
        #Compute the CDF given in equation (17)
        demand_mean=demand_mean+1/std*(Y[i]-demand_mean[i])*demand_cov[:,i]
        demand_mean=np.delete(demand_mean,index[i])
        demand_cov =np.delete(np.delete(demand_cov -1/std *np.matmul(demand_cov[:i].T,demand_cov[:i]),(i),axis=0),(i),axis=1)
        #small note here is that the demand[1] is just to test if it works otherwise its std but for some reasons it was an overflow error (std=0 appearantly)
        partial_F =normpdf(Y[j],demand_mean[j],demand_mean[1]) *mvn(mean=demand_mean, cov=demand_cov).cdf(np.array[Y].delete(index[j])) #
        
    return np.array(list(partial_F))
"""
def multi_gradient(samples,Y):
    
    '''
    function that computes the gradient of \alpha -F(Y) 
    Parameters
    ----------
    Y : TYPE     numpy.array or pars['Y'] as explained in util
        DESCRIPTION. you insert an array Y of shape (1,8) 
    Returns
    Samples: from util
    -------
    None.
    
    '''
    demands, demand_mean, demand_cov = samples
    #partial_F=np.zeros((1,8))
    partial_F=[]
    std=np.sqrt(np.diag(demand_cov))
    index=[0,1,2,3,4,5,6,7]
    for i, j in product(range(8),range(8)):
        #Compute the CDF given in equation (17)
        neew_mean=new_mean+1/std*(Y[i]-new_mean[i])*new_cov[:,i]
        it_mean=np.delete(new_mean,index[i])
        new_cov =np.delete(np.delete(new_cov -1/std *np.matmul(new_cov[:i],new_cov[:i].T),(i),axis=0),(i),axis=1)
        partial_F =mvn(mean=new_mean, cov=new_cov).cdf(np.array[Y].delete(index[j])) *normpdf(Y[j],new_mean[j],std) #
        
    return np.array(list(partial_F))
"""

# this stuff should be in the main.py
np.set_printoptions(precision=3, suppress=False)
args = util.parse_args()
pars = util.get_parameters()
demands = draw_new_samples(args, pars)
ob1,sol1,Y_bar,delta= run_LP_11(demands, pars)
ob2,sol1_K,Y_0,delta_k= run_LP_11_K(demands,pars)

#Y=Y_bar['Y-'].reshape(-1)
#print(g=multi_gradient(demands,Y))
#use solution of the above LP as the interior point for starting the line search and iteration
"""
Y_bar=sol2['Y'] #interior point from solution of LP_11
Y_0=sol2['Y']   #boundary point from solution of LP_11_K
test the value of F(Y_0)<0.5 (it should be) error I get is:
    array 'mean' must be a vector of lenght 8
mvn(mean=new_mean, cov=new_cov).cdf(np.array[Y_0]) 

#Iterate for adding cuts 
"""
maxiter=20
for it in range(maxiter):
    ob1,sol1,Y_ba,delta= run_LP_11(demands, pars)
    ob2,sol1_K,Y_0,delta_k= run_LP_11_K(demands,pars)
    Y_bar=Y_bar['Y-'].reshape(-1)
    Y_0=Y_0['Y-'].reshape(-1)
    if multi_gradient(samples, Y_bar)*multi_gradient(samples, Y_0)>=0:
        break
    print("bisection failed one of them is the solution")
    while multi_gradient(samples, Y_bar)*multi_gradient(samples, Y_0)<0:
        Y_0=0.5Y_0+0.5*Y_bar #bisection
        #add cuts and solve the linear programm LP_11_K for the smaller feasible set
        
        
    print('Maximal iteration', it)
    #print('Optimal objective value', LP_11_K('obj'),'\nY_bar', sol2[Y-],...)
     
