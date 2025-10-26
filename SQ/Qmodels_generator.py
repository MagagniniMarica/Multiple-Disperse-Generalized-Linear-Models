# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:01:34 2025

@author: Maric
"""

from pyomo import environ as pym
from Funs import Dataset_selection
import pandas as pd
import numpy as np

def np_ceil_dec(x, decimali=2):
    fattore = 10 ** decimali
    return np.ceil(x * fattore) / fattore

# 
# GML : Linear Regression (Lin), Logistic Regression (Log), Poisson Regression (Poi)
#
GLM_list = ['Lin']#, 'Log', 'Poi'

Q=1

for GLM in GLM_list:
    #
    # Data    
    #
    dataset, target, features, features_type = Dataset_selection(GLM)
    N,J = dataset.shape

    
    Solver = {'Lin': 'gurobi_persistent', 'Log':'ipopt', 'Poi':'ipopt'}.get(GLM)
    ########################################################################
    # Model
    ########################################################################
    #
    # Model definition
    #
    m = pym.ConcreteModel(name = 'Q model')
    
    #
    # Indexes
    #
    m.j = pym.Set(initialize = features)                     # All features
    features_bias = features.append(pd.Index(['bias']))
    m.j_b0 = pym.Set(initialize = features_bias)             # All features and bias
    
    m.q = pym.RangeSet(0,Q-1)
    
    # Number of instances
    N =  len(dataset)   
    m.n = pym.RangeSet(0,N-1)
    
    # Target variable
    m.y = pym.Param(m.n, initialize=target)
    
    # Dataset with exceeding column of 1s that refers to the bias term
    dataset_bias = dataset.copy()
    dataset_bias['bias'] = np.ones(len(dataset))
    def x_init(m,n,j):
        return dataset_bias.iloc[n][j]
    m.x = pym.Param(m.n,m.j_b0, initialize=x_init, mutable=True)
    
    #
    # variables
    #
    m.beta = pym.Var(m.q,m.j_b0, within=pym.Reals, bounds= (-30,30))
    
    #
    # Objective function 
    #
    
    def beta_xn_(m, n,q):
        return sum(m.beta[q, j] * m.x[n, j] for j in m.j_b0)
    
    def objfunction_(m, GLM):
        if GLM == 'Lin':
            return   sum( (1/N) *sum((m.y[n] - beta_xn_(m, n,q))**2 for n in m.n) for q in m.q)
        
        elif GLM == 'Log':    
            # Questa approssimazione l'abbiamo fatta perché sennò da problemi numerici 
            # M =10e15
            # return -(1 / N) * sum( beta_xn_(m, n,q)*(m.y[n] - 1) 
            #                       - pym.log(1 + pym.exp(-beta_xn_(m, n,q))) if beta_xn_(m, n,q) > -M else
            #                       beta_xn_(m, n,q)*(m.y[n])
            #                       for n in m.n for q in m.q)
        
            # senza approssimazione ma stabile
            return (1 / N) * sum( (1 - m.y[n]) * beta_xn_(m, n, q) + 
                                 pym.log(1 + pym.exp(-beta_xn_(m, n, q))) 
                                 for n in m.n for q in m.q )
            # Senza approssimazione 
            # return -(1 / N) * sum( beta_xn_(m, n,q)*(m.y[n] - 1) - pym.log(1 + pym.exp(-beta_xn_(m, n,q)))  for n in m.n for q in m.q)
        
        elif GLM == 'Poi':
            return sum( pym.exp(beta_xn_(m, n,q)) - beta_xn_(m, n,q)*(m.y[n])  for n in m.n for q in m.q)
            


         
    m.objfunction = pym.Objective(rule=objfunction_(m,GLM),  sense=pym.minimize)
    
    ######################################################################
    #
    ######################################################################
    
    # 
    # Solver
    #

    solver = pym.SolverFactory(Solver)
    # solver.options['TimeLimit'] =180
    # solver.options['MIPFocus'] = 1
    # solver.options['NumericFocus'] = 2
    
    # Collega il modello all'istanza del solver
    solver.set_instance(m)
    
    #
    # Results
    #
    #result = 
    solver.solve(tee = True)
    
   
    
    #
    # Print Results
    #
    print("Input Model Q")
    betaQ = []
    for q in m.q:
        print('Model %d)' % (q+1))

        betaQ_q= []
        for f in m.j_b0:
            betaQ_q.append(m.beta[q,f].value)
            print(f"beta[{q},{f}] =", m.beta[q, f].value)
            
        print("\n")
        betaQ.append(betaQ_q) # Model obtained 
    
    # Recupera il primo (e di solito unico) obiettivo attivo del modello
    for obj in m.component_objects(pym.Objective, active=True):
        tau_min = pym.value(obj) 
        print(f"Current obj value ($tau$ min): {pym.value(obj)}")
        
    #
    # Save 
    #
    
    # Crea DataFrame e salva
    df = pd.DataFrame(betaQ, columns=m.j_b0.data())
    df['tau_min'] = np_ceil_dec(tau_min, decimali=2) 
    full_path =f'C:/Users/Maric/Dropbox/Multiple disperse GLMs/Experiments/SQ/SQ_{GLM}.xlsx'
    df.to_excel(full_path, index=False)

    print(f"File coeff salvato come 'SQ_{GLM}.xlsx' in {full_path}")
    
"""
Altra opzione per scrivere una loss stabile

from pyomo.environ import log, exp, max_

def logistic_loss(z, y):
    # y in {0, 1}
    return max_(0, -z) + log(1 + exp(-abs(z))) if y == 1 else max_(0, z) + log(1 + exp(-abs(z)))
"""    