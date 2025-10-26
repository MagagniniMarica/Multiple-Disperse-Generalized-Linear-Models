# -*- coding: utf-8 -*-
"""

@author: Marica Magagnini

Fixed tau (accuracy constraint) and theta (feature selected), this problem 
compute P GLMa maximizing the dispersion. 

The combinatorial part of this problem is managed by a VNS strategy.

"""
from pyomo import environ as pym
import numpy as np
import pandas as pd

# dispersion  = 'l1','l2', 'o1'
# GLM = Lin, Log, Poi
# FSP feature selection input provided by vsn
#'l1' --> (T,E) Combinatorial Structure
#'o1' --> (V,W) Combinatorial Structure
def P_max_disp_local_(GLM,dispersion, dataset, target, features, FSP,
                          SQ, P, tau, X0, 
                          V = None,W = None, T=None, E=None):
    
    #
    # Model definition
    #
    m = pym.ConcreteModel(name = f'Local D-PDP-{GLM}-{dispersion}')

    #
    # Indexes
    #
    m.j = pym.Set(initialize = features)                     # All features
    features_bias = pd.Index(['bias']).append(features)
    m.j_b0 = pym.Set(initialize = features_bias)             # All features and bias
    
    J = len(features)
    J1 = J+1
    
    # Number of instances
    N =  len(dataset)   
    m.n = pym.Set(initialize=dataset.index.tolist())
    
    # Number of known regressors
    Q = len(SQ)
    m.q = pym.RangeSet(0,Q-1)
    
    #Number of new regressors
    m.p = pym.RangeSet(0,P-1)
    
    #
    # Parameters
    #
    
    # Target variable
    m.y = pym.Param(m.n, initialize=target)
    
    # Dataset with exceeding column of 1s that refers to the bias term
    dataset_bias = dataset.copy()
    dataset_bias['bias'] = np.ones(len(dataset))
    def x_init(m,n,j):
        return dataset_bias.iloc[n][j]
    m.x = pym.Param(m.n,m.j_b0, initialize=x_init, mutable=True)
    
    
    # If the dispersion is on the response, we need to initialize  X^0 
    ############   
    if  dispersion == 'o1':
        m.X0  = pym.Param(m.j_b0, initialize=X0)
    ############   
    
    if dispersion  == 'o1':
        def v_init(m,p,p_prime):
            return V[p,p_prime]
        m.v = pym.Param(m.p, m.p, initialize=v_init, mutable=True)
        def w_init(m,p,q):
            return W[p,q]
        m.w = pym.Param(m.p, m.q, initialize=w_init, mutable=True)
     

    if dispersion  == 'l1':

        def e_init(m,j,p,p_prime):
            return E[j][p,p_prime]
        m.e = pym.Param(m.j, m.p,m.p,initialize=e_init,mutable=True)

        def t_init(m,j,p,q):
            return T[j][p,q]
        m.t = pym.Param(m.j,m.p,m.q,initialize=t_init,mutable=True)
    
    
    
    
    # beta coefficients of known regressors 
    def SQ_init(m,q,j):
        return SQ.iloc[q][j]
    m.SQ = pym.Param(m.q,m.j_b0, initialize=SQ_init,mutable=False)
    
    
    #
    # variables
    #
    m.obj =  pym.Var(within=pym.NonNegativeReals) 
    m.beta = pym.Var(m.p, m.j_b0, within=pym.Reals, bounds= (-30,30))
    

    # Fix to zero some variables as a result of the combinatorial part provided by the vns
    for p in range(len(FSP)):
        for f_j, f in zip(range(1,J1),features):
            if FSP[p][f_j] == 0:
                m.beta[p, f].fix(0)
              
    #
    # Objective function 
    #
    def objfunction_(m):
        return m.obj
    m.objfunction = pym.Objective(rule=objfunction_,  sense=pym.maximize)
    
    
    
    #
    # Initialize constraints
    #

    if dispersion == 'l1':
        def c_extra_dispersion_1_(m, p,q):
            expr = 0
            for j in m.j:
                if m.t[j,p, q].value != 0:
                    expr += ( m.SQ[q, j] - m.beta[p, j]) * m.t[j,p, q]
                else:
                    expr += abs(m.SQ[q, j])
            return expr >= m.obj
        
        def c_extra_dispersion_2_(m,j,p,q):
            if m.t[j,p, q].value != 0:
                return (m.beta[p,j] - m.SQ[q,j]) * m.t[j,p,q] <= 0
            else: 
                return pym.Constraint.Skip 
        
        def c_intra_dispersion_1_(m, p, p_prime):
            if p < p_prime:
                return sum((m.beta[p_prime,j] - m.beta[p,j]) * m.e[j,p,p_prime] for j in m.j) >= m.obj
            else: 
                return pym.Constraint.Skip 
        
        
        def c_intra_dispersion_2_(m,j, p, p_prime,):

            if p < p_prime and m.e[j,p,p_prime].value !=0:
                return  (m.beta[p,j] - m.beta[p_prime,j])*m.e[j,p,p_prime] <=0
            
            else: 
                return pym.Constraint.Skip 
        

    elif dispersion == 'l2':
       
        def c_extra_dispersion(m,q,p):
            return pym.quicksum((m.SQ[q,j] - m.beta[p,j])**2 for j in m.j ) >= m.obj**2
        
        def c_intra_dispersion(m,p,p_prime):
            if p < p_prime:
                return pym.quicksum((m.beta[p,j] - m.beta[p_prime,j])**2 for j in m.j ) >= m.obj**2
            else: 
                return pym.Constraint.Skip
            
         
    elif dispersion == 'o1':
        
           
        def c_intra_dispersion_(m, p, p_prime):
            if p < p_prime:
                return (pym.quicksum(m.beta[p,j]*m.X0[j]   for j in m.j_b0) -  pym.quicksum(m.beta[p_prime,j]*m.X0[j] for j in m.j_b0)) * (2*m.v[p,p_prime]-1) >= m.obj
            else: 
                return pym.Constraint.Skip
        
        def c_extra_dispersion_(m, p,q):
            return ( pym.quicksum(m.beta[p,j]*m.X0[j] for j in m.j_b0)- pym.quicksum(m.SQ[q,j]*m.X0[j]   for j in m.j_b0)) * (2*m.w[p,q] - 1) >= m.obj
   
    elif dispersion == 'dsa':
        pass
             
    else:
        print('Dispersion norm not recognized. ')
        
        
        
        
    #Accuracy constraint    
    def beta_xn_(m, n,p):
        return sum(m.beta[p, j] * m.x[n, j] for j in m.j_b0)
     
         
    def c_accuracy_p_(m, p):
        if GLM == 'Lin': 
            return sum( (m.y[n] - beta_xn_(m, n,p) )**2 for n in m.n) <= N*(tau)
        elif GLM == 'Log':    
            return sum( (1 - m.y[n]) * beta_xn_(m, n,p) + pym.log(1 + pym.exp(-beta_xn_(m, n,p))) 
                                 for n in m.n ) <= N*tau
        
        elif GLM == 'Poi':
            return sum( pym.exp(beta_xn_(m, n,p)) - beta_xn_(m, n,p)*(m.y[n])  for n in m.n) <= tau
             
    #
    # Declare constraints
    #
    
    if dispersion == 'l1':

        m.c_Edis1 = pym.Constraint(m.p, m.q, rule = c_extra_dispersion_1_)
        m.c_Edis2 = pym.Constraint(m.j, m.p, m.q, rule = c_extra_dispersion_2_)
        m.c_Idis1 = pym.Constraint(m.p, m.p, rule = c_intra_dispersion_1_)
        m.c_Idis2 = pym.Constraint(m.j, m.p, m.p,  rule = c_intra_dispersion_2_)
       
       
    elif dispersion == 'l2':
        m.c_Edis = pym.Constraint(m.q, m.p, rule = c_extra_dispersion)
        m.c_Idis = pym.Constraint(m.p, m.p, rule = c_intra_dispersion)
    
    elif dispersion == 'o1':
     
        m.c_Edis = pym.Constraint(m.p, m.q, rule = c_extra_dispersion_)
        m.c_Idis = pym.Constraint(m.p, m.p, rule = c_intra_dispersion_)
               

    m.c_acc_p_ = pym.Constraint(m.p, rule = c_accuracy_p_)

    
    return m