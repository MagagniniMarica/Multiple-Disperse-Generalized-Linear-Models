# -*- coding: utf-8 -*-
"""

@author: Marica Magagnini

This file contains the main function to perform the experiments of the project
"Multiple dipserse GLM."

Most function can be employed in both of the two cases:
    - maximizing the accuracy. 
    - maximazin the dispersion.
    
Case-specific function has an appropriate name. 
"""

import pandas as pd
import os
from pyomo import environ as pym
from f_print import get_obj_sol_
import gurobipy as GRB

#
# User path selection
#
#
User = '../...' #<--- Fill in with yoru own path

#
#  Dataset
# 
def Dataset_selection(GLM):
    """
    This function select the dataset for the specific GLM:
        - Linear Regression 'Lin': Boston Housing dataset with continuous target values.
        - Logistic Regression 'Log': Boston Housing dataset with binary target values (0-1).
        - Poisson Regression 'Poi':  Seoul Bike Sharing Demand with positive integer values
    """
    
    if GLM == 'Lin':
        from BostonHousing import data
        task = ''
        path = f'C:/{User}/Datasets/Boston Housing/'
    elif GLM == 'Log':
        from BostonHousing import data
        task = 'classification'
        path =f'C:/{User}/Datasets/Boston Housing/'
    elif GLM == 'Poi':
        from SeoulBike import data
        task = ''
        path = f'C:/{User}/Datasets/SeoulBike/'
    
    return data(path,task)

#
# Known models
#
def S_Q_(GLM, features):
    """
    
    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    features : array of strings
        Features names of the dateset corresponding to the selectes GLM.

    Returns
    -------
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    SQ_obj : Series
        It contains the accuracy of the Q models.

    """
    full_path =f'C:/{User}/SQ/SQ_{GLM}.xlsx'
    df = pd.read_excel(full_path)
    
    SQ_obj = df['tau_min']
    subset = ['bias'] + list(features)
    SQ = df[subset]
    
    return SQ, SQ_obj


###############################################################################
# Maximize dispersion (D-PDP-GLM): call and solve a specific instance
###############################################################################
    
def PCD_problem_MAXDISP_(GLM, Solver_,
                 dataset,target, 
                 features, SQ,P, tau, theta, M, X0,
                 dispersion, runs_vns,TimeLimit_vns,
                 SP_feas, gamma_max_feas):
    """
    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt" 
    dataset : dataframe
        Only feature values.
    target : Series
        Target values of the dataset.
    features : array of strings
        FEtures names.
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    P : int
        Number of models to be built.
    tau : float
        accuracy threshold.
    theta : int
        Number non-zero features in the new P-models.
    M : int
        Big-M parameter.
    X0 : Series or dataframe
        Selected instance of the dataset used in case of 'o1/o2' dispersion. 
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    runs_vns : int
        Heuristic multi-start parameter: Number of times to restart the VNS 
    TimeLimit_vns : int
        VNS time-limit for a sing run. 

    Returns
    -------
    SP : disct of dict
        It containts P-dicts, where each one is a new model. Each dict contains 
        the model coefficients (bias included). 
    SP_obj : float
        Maximal dispepersion computed for the set of P-models.
    status : String or NoneType
        It containts the gurobi final status when it is used as solver. 

    """
    
    N,J = dataset.shape     # N: number of dataset elements, number of features
    J1 = J+1                # number of fetures + bias term
    status = None
    
    ###########################################################################
    # Linear Regression - Exact computation
    ###########################################################################
    if GLM == 'Lin' and Solver_ == 'gurobi_persistent':       
        
        #
        # Instance call
        #
        from P_CD_problem_MaxDisp.PCD_MAX_dispersion_Lin import P_max_dispersion_Lin_
        instance = P_max_dispersion_Lin_(dispersion, dataset, target, features, M, 
                         SQ, P, tau, theta, X0)
        
        # 
        # Solver
        #
        solver = pym.SolverFactory(Solver_)
        solver.set_instance(instance)
        
        #Solver parameters
        solver.options['TimeLimit'] =1800
        
        #
        # Results
        #
        solver.solve(tee = True)
        
        #
        # Solver status
        #
        
        grb_model = solver._solver_model

        # Status and number of solutions
        status_code = grb_model.Status
        sol_count = grb_model.SolCount if hasattr(grb_model, "SolCount") else 0

        # Obj and MIPGap
        if sol_count > 0 and grb_model.SolCount > 0:
            objval = grb_model.ObjVal
            mipgap = grb_model.MIPGap if hasattr(grb_model, "MIPGap") else float('nan')
            
            global get_obj_sol_
            SP_obj, SP = get_obj_sol_(instance)
        else:
            objval = float('nan')
            mipgap = float('nan')
            
            SP, SP_obj = ([],0)

        # Status Message
        status_msg = {
            GRB.GRB.OPTIMAL: "Optimal Solution Found",
            GRB.GRB.TIME_LIMIT: "Time limit reached",
            GRB.GRB.INFEASIBLE: "Infeasible Problem",
            GRB.GRB.UNBOUNDED: "Unbounded Problem",
            }.get(status_code, f"Unknown status (code={status_code})")

        status = (objval,mipgap,status_msg)
   
    ###########################################################################
    # Linear, Logistic,  Poisson Regression - Heuristic Results
    ###########################################################################
    else:
        # Select local solver maximum number of iteration 
        solver_Iterations = {'Lin': 200, 'Log': 200, 'Poi': 1000}.get(GLM, None)
        
        # Call heuristic strategy
        from P_CD_problem_MaxDisp.VNS_MaxDisp import VNS, VNS_dsa_
        
        
        #
        # VNS parameters
        #
        K = J1-theta 


        SP, SP_obj = (SP_feas, gamma_max_feas)
        status = 'FAIL : Heuristic did NOT find a solution better than the feasible initial one.'
        
        
        for run in range(runs_vns):
            # dispersion 'dsa' has a slightly different VNS strategy. 
            
            if dispersion == 'dsa':
                RES, FSPvns, betaPvns, obj_vns = VNS_dsa_(GLM, Solver_, 
                                                          K,TimeLimit_vns,
                                                          dataset, target, features, SQ, 
                                                          P,J1,theta, tau, 
                                                          X0,solver_Iterations)
                
            else:
                RES, FSPvns, betaPvns, obj_vns = VNS(GLM, Solver_,
                                                     K,TimeLimit_vns,
                                                     dataset, target, features, SQ, 
                                                     P,J1,theta, tau,dispersion, 
                                                     X0, solver_Iterations)
            
            
            # Select the best result among all the VNS runs (largest dispersion value)
            if obj_vns > SP_obj: 
                SP_obj = obj_vns         
                SP = betaPvns    
                status = 'SUCCESS : Heuristic FOUND a solution better than the feasible initial one.'
                

            
    return SP, SP_obj, status

###############################################################################
# Maximize accuracy (A-PDP-GLM): call and solve a specific instance
###############################################################################


# Parameters for Max Acc case
def params_(GLM, dispersion):
    """
    

    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'

    Returns
    -------
    parameters : dataframe
        Columns: tau (accuracy), theta (number feature selected), gamma_max 
                (maximal dipersion computed for the couple (tau,theta)).

    """
    
    # Build File name
    directory = f'C:/{User}/A/params/'
    nome_file = f"D_Obj_{GLM}_{dispersion}.xlsx"
    percorso_completo = os.path.join(directory, nome_file)

    # Check
    if not os.path.exists(percorso_completo):
        print(f"File non trovato: {percorso_completo}")
        return

    # Read Data
    parameters = pd.read_excel(percorso_completo)

    
    return parameters


def PCD_problem_MAXACC_(GLM, Solver_,
                 dataset,target,
                 features, SQ, P, gamma, tau, theta, M, X0,
                 dispersion, runs_vns,TimeLimit_vns,
                 SP_feas, obj_feas):
    """

    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt" 
    dataset : dataframe
        Only feature values.
    target : Series
        Target values of the dataset.
    features : array of strings
        FEtures names.
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    P : int
        Number of models to be built.
    gamma : float (or int in case of dispersion = 'dsa')
        dispersion threshold.
    tau : float
        accuracy threshold.
    theta : int
        Number non-zero features in the new P-models.
    M : int
        Big-M parameter.
    X0 : Series or dataframe
        Selected instance of the dataset used in case of 'o1/o2' dispersion. 
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    runs_vns : int
        Heuristic multi-start parameter: Number of times to restart the VNS 
    TimeLimit_vns : int
        VNS time-limit for a sing run. 


    Returns
    -------
    SP : dict of dict
        It containts P-dicts, where each one is a new model. Each dict contains 
        the model coefficients (bias included). 
    SP_obj : float
        Maximal accuracy (min error/los) computed for the set of P-models.
    status : String or NoneType
        It containts the gurobi final status when it is used as solver.     

    """
    
    
    N,J = dataset.shape
    J1 = J+1
    status = None
    
    ###########################################################################
    # Linear Regression - Exact computation
    ###########################################################################
    if GLM == 'Lin' and Solver_ == 'gurobi_persistent':       
        
        #
        # Instance call
        #
        from P_CD_problem_MaxAcc.PCD_Max_Acc_Lin import P_cond_disp_
        instance = P_cond_disp_(dispersion, dataset, target, features, 
                                  SQ, P, gamma, tau, theta, M, X0)
        # 
        # Solver
        #
        solver = pym.SolverFactory(Solver_)
        # Collega il modello all'istanza del solver
        solver.set_instance(instance)
               
        #Solver parameters
        solver.options['TimeLimit'] =300

        
        #
        # Results
        #
        solver.solve(tee = True)
        
        #
        # Solver status
        #
        
        # Accesso al modello Gurobi nativo
        grb_model = solver._solver_model

        # Stato e numero di soluzioni
        status_code = grb_model.Status
        sol_count = grb_model.SolCount if hasattr(grb_model, "SolCount") else 0

        # Valore funzione obiettivo e MIPGap, se disponibile
        if sol_count > 0 and grb_model.SolCount > 0:
            objval = grb_model.ObjVal
            mipgap = grb_model.MIPGap if hasattr(grb_model, "MIPGap") else float('nan')
            
            global get_obj_sol_
            SP_obj, SP = get_obj_sol_(instance)
        else:
            objval = float('nan')
            mipgap = float('nan')
            
            SP, SP_obj = ([],0)

        ## Status Message
        status_msg = {
            GRB.GRB.OPTIMAL: "Optimal Solution Found",
            GRB.GRB.TIME_LIMIT: "Time limit reached",
            GRB.GRB.INFEASIBLE: "Infeasible Problem",
            GRB.GRB.UNBOUNDED: "Unbounded Problem",
            }.get(status_code, f"Unknown status (code={status_code})")

        status = (objval,mipgap,status_msg)
            
            
        print("This is an optimal solution.")

    ###########################################################################
    # Linear, Logistic,  Poisson Regression - Heuristic Results
    ###########################################################################
    else:
        # Select local solver maximum number of iteration 
        solver_Iterations = {'Lin': 200, 'Log': 200, 'Poi': 1000}.get(GLM, None)
        
        from P_CD_problem_MaxAcc.VNS_Max_Acc import VNS

        #
        # VNS parameters
        #
        K = J1-theta
        
        SP, SP_obj = (SP_feas, obj_feas)
        status = 'FAIL : Heuristic did NOT find a solution better than the feasible initial one.'
        
        
        for run in range(runs_vns):
            
            RES, FSPvns, betaPvns, obj_vns = VNS(GLM, Solver_, 
                                                 K,TimeLimit_vns,  
                                                 dataset, target, features, SQ,
                                                 P,J1,theta, gamma, tau,dispersion, 
                                                 X0, solver_Iterations)
            
            # SP_obj best among all the heuristic runs
            if obj_vns < SP_obj: 
                SP_obj = obj_vns         
                SP = betaPvns               # SP of of all the runs
                status = 'SUCCESS : Heuristic FOUND a solution better than the feasible initial one.'
            
        
    return SP, SP_obj, status


###############################################################################
# Save Results
###############################################################################

# P-models Coefficients
def save_results_coeff_xlsx(AD, Solver_, GLM, dispersion, theta, tau, gamma, 
                            SP,SP_obj, SQ, SQ_obj,  features, filename = None):
    """
    

    Parameters
    ----------
    AD : string
        'D' - maximal dispersion 
        'A' - maximal accuracy
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt". 
        It defines the type of solution -- > 'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    tau : float
        accuracy threshold.
    theta : int
        Number non-zero features in the new P-models.
    gamma : float
        Dipersion lower bound (only when AD = 'A')
    SP : disct of dict
        It containts P-dicts, where each one is a new model. Each dict contains 
        the model coefficients (bias included). 
    SP_obj : float
        If AD = 'D' :  Maximal dispepersion computed for the set of P-models.
        If AD = 'A' :  Maximal accuracy (sum (or mean?)) computed for the set of P-models.
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    SQ_obj : Series
        It contains the accuracy of the Q models.
    features : array of strings
        Features names of the dateset corresponding to the selectes GLM.
    filename : TYPE, optional
        Name of the file to create. The default is None.
        

    Returns
    -------
    Save results in a .xlsx file. 
    The first set of P rows containts the results of the set of models 
    built for a spacific combination of parameters. 
    The second set of Q models are the already known models.
    File name framework:
        filename = f"{AD}_Coeff_{Type}_{GLM}_{dispersion}_tau{tau}_theta{theta}_gamma{gamma}.xlsx"
    
    Directory: {AD}_{Type}_{GLM}_{dispersion}
    
    The 'Obj' column is :
        - the maximal dispersion (or gamma_max) when AD = 'D';
        - the maximal accuracy when AD = 'A'.

    """
    #
    # Build Directory 
    #
    Type = {'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'}.get(Solver_, None)
    save_dir = os.path.join('.', f'{AD}_{Type}_{GLM}_{dispersion}')
    os.makedirs(save_dir, exist_ok=True)
    
    #
    # Filename if not provided
    #
    if filename is None:
        filename = f"{AD}_Coeff_{Type}_{GLM}_{dispersion}_tau{tau}_theta{theta}_gamma{gamma}.xlsx"
       
    full_path = os.path.join(save_dir, filename)

    
  
    dati = []

   
    # Append SP models with the complete obj function (sum of the P)        
    P = len(SP)
    for p in range(P):
        row = [p+1]+ [SP[p][f] for f in   ['bias'] + list(features)] + [SP_obj]
        dati.append(row)
    
    # Append SQ models
    Q = SQ.shape[0]
    for q in range(Q):
        row = [p+2] + [SQ.iloc[q][f].item() for f in   ['bias'] + list(features)] +[SQ_obj[q]]
        dati.append(row)
    

    # Columns names
    colonne = [f'{GLM}_P1:{P}_Q{P+1}:']+ ['bias'] + list(features)  + ['Obj']
    
    
    # Save dataframe to .xlsx
    df = pd.DataFrame(dati, columns=colonne)
    df.to_excel(full_path, index=False)

    print(f"File coeff salvato come '{filename}' in {full_path}")
    
    
#Objective values   
def save_objRes_xlsx(AD, Solver_, GLM,dispersion, parms_objs, filename = None):
    """

    Parameters
    ----------
    AD : string
        'D' - maximal dispersion 
        'A' - maximal accuracy
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt". 
        It defines the type of solution -- > 'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    parms_objs : tuple 
        If AD = 'D' : (tau, theta, obj) where obj is the maximal dispersion
        If AD = 'A' : (tau, theta, gamma, obj) where obj is the maximal accuracy
    filename : TYPE, optional
        Name of the file to create. The default is None.

    Returns
    -------
    Save results in a .xlsx file. 
    Directory: {AD}_{Type}_{GLM}_{dispersion}
    File name framework:
        filename = f"{AD}_Obj_{GLM}_{dispersion}.xlsx"
        
    Each row represnts a different experiments where the parameters are record in the 
    firsts columns. Last column is the value of the objective function. 
        

    """
  
    #
    # Build Directory 
    #
    Type = {'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'}.get(Solver_, None)
    save_dir = os.path.join('.', f'{AD}_{Type}_{GLM}_{dispersion}')
    os.makedirs(save_dir, exist_ok=True)
    
    
    #
    # Filename if not provided
    #
    if filename is None:
        filename = f"{AD}_Obj_{GLM}_{dispersion}.xlsx"
    
    full_path = os.path.join(save_dir, filename)

    
    # Build Dataframe
    if AD == 'A':
        df = pd.DataFrame(parms_objs, columns=['tau','theta', 'gamma', 'SP_obj'])
    if AD == 'D':
        df = pd.DataFrame(parms_objs, columns=['tau','theta', 'gamma_max'])
   
        
    # Save dataframe to .xlsx
    df.to_excel(full_path, index=False)
    print(f"File objs salvato come '{filename}'")

   
   