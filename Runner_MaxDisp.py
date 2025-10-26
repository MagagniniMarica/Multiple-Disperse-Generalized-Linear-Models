# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini

This is the main file to solve a set of instances of the (D-PDP-GLM) problem, 
i.e. maximum  dispersion problem. 
"""
#
# Functions to run the experiments
#
from Funs import Dataset_selection, S_Q_, PCD_problem_MAXDISP_, save_results_coeff_xlsx, save_objRes_xlsx



# 
# GML : Linear Regression (Lin), Logistic Regression (Log), Poisson Regression (Poi)
#
GLM_list = [ 'Lin','Log','Poi'] 



#
# Dispersion Notion: 'l1', 'l2', 'dsa', 'o1' 
# 
Dispersion_list = [ 'dsa', 'o1', 'l1', 'l2']


#
# Solver optimal or local ( Note that gurobi solves only the linear case)
#
Solvers = ['gurobi_persistent', 'ipopt'] 


#
# P-CD_GLM Parameters
#
P = 3                               # Number of new GLMs
M = 100                             # big M for linear case 

THETA  = [6,7,8,9]
TAU_percentage = [0.1, 0.15, 0.2]   

#%% Funtion to retrive a feasible starting solution, if there is from previous executions.

def feas_sol(theta, t_pc, pool):
    """
    

    Parameters
    ----------
    theta : int
        sparsity  in [6,7,8,9]
    t_pc : float
        accuracy percentage in [0.1,0.15,0.2]
    pool : list of tuples
        List of the results of other instances of the problem: 
            pool[(theta,t_pc)] = {
                "gamma_max": gamma_max,
                "SP": SP
            }

    Returns
    -------
    If available it returns a feasible solution for the instance of the problem 
    to be solved
    
    SP_feas : list of models
    gamma_max_feas : float

    """
    
    # Feasible solution, alreasy known, for the specific instanc of the max dispersion 
    # problem. No feasibile solution is knonw for the first case, i.e. theta = 6, t_pc = 0.1
    SP_feas, gamma_max_feas = (None,None)
    
    
    if theta == 6 and t_pc == 0.1:
        SP_feas, gamma_max_feas = ([],0)
    elif theta == 6 and t_pc > 0.1:
        theta_feas = theta
        t_pc_feas = round(t_pc - 0.05, 2)
    
    elif theta > 6 and t_pc == 0.1:
        theta_feas = theta -1
        t_pc_feas = t_pc 

    
    else: #theta > 6 and t_pc > 0.1
        # choose the one with highest gamma_max between (theta-1, t_pc) and (theta, t_pc-0.05)
        try:
            option_1 = pool[(theta-1, t_pc)]['gamma_max']
        except:
            option_1  = 0
        try:
            option_2 = pool[(theta,  round(t_pc - 0.05, 2))]['gamma_max']
        except:
            option_2  = 0
        
        if option_1 > option_2:
            theta_feas = theta -1
            t_pc_feas = t_pc 
        else:
            theta_feas = theta
            t_pc_feas =  round(t_pc - 0.05, 2)
    
    if gamma_max_feas == None:
        try:
            SP_feas = pool[(theta_feas,t_pc_feas)]['SP']
            gamma_max_feas = pool[(theta_feas,t_pc_feas)]['gamma_max']
        except:
            print('not found')
            SP_feas, gamma_max_feas = ([],0)
        
    return SP_feas, gamma_max_feas

#%% Executions

#
# Multi-start approach for the heuristic
#
runs_vns = 5 


for GLM, Solver_ in [('Lin', Solvers[0]), ('Lin', Solvers[1]), ('Log', Solvers[1]), ('Poi', Solvers[1])]:    
    #
    # Data    
    #
    dataset, target, features, features_type = Dataset_selection(GLM)
    N,J = dataset.shape


    X0 = dataset.iloc[0]
    X0["bias"] = 1
    X0 = X0[["bias"] + [col for col in X0.index if col != "bias"]]

    #
    # Input model
    #
    SQ, tau_min = S_Q_(GLM, features)  
    
    TimeLimit_vns = {'Lin': 120, 'Log': 120, 'Poi': 180}.get(GLM, None)     


    for dispersion in Dispersion_list:
        
        if Solver_ == 'gurobi_persistent':
            logfile = open(f"D_log_solver_status_{dispersion}.txt", "a")
        else:
            logfile = open(f"D_heur_status_{dispersion}.txt", "a")
        
        print('---------------------------------------------------------------------------')
        print(f'GLM: {GLM}, Solver : {Solver_}, dispersion: {dispersion}. N = {N}, J = {J}')
        objs = []
        
        # Indexed by theta and t_pc, contains SP and gamma_max
        pool = {} 
        
    
        
        # theta: number of feture selected 
        for theta in THETA:
            # accuracy percentage requirement
            for t_pc in TAU_percentage: 
                

                print(f'{GLM}, {dispersion}, {t_pc}')
                # accuracy bound
                tau = tau_min.item() + t_pc*abs(tau_min.item())
                print(f' tau :{tau}, theta: {theta} (+{t_pc*100}% tau_min)')
                
                # Fesible solution 
                SP_feas, gamma_max_feas = feas_sol(theta, t_pc, pool)
                
                
                #
                # Call and solve the instance
                #
                SP, gamma_max, status = PCD_problem_MAXDISP_(GLM, Solver_,
                                                  dataset,target, 
                                                  features, SQ,P, tau, theta, M, X0,
                                                  dispersion, runs_vns,TimeLimit_vns,
                                                  SP_feas, gamma_max_feas)
                
                
                if SP != []:
                    # Save Instance Results: P-models coefficients save
                    save_results_coeff_xlsx('D',Solver_, GLM, dispersion, theta,tau, gamma_max, 
                                            SP,gamma_max, SQ,[0],  features, filename = None)
                
                # Record current intance best dispertion found
                objs.append((tau, theta, gamma_max))
                
                # Record all SP and gamma_max associated to theta and t_pc
                
                pool[(theta,t_pc)] = {
                    "gamma_max": gamma_max,
                    "SP": SP
                }
                
                #
                #  Write log  -- Solver/heristic status
                #
                logfile.write(f"GLM={GLM}, Dispersion={dispersion}, tau={tau:.5f}, theta={theta}, ")
                if Solver_ == 'gurobi_persistent':
                    logfile.write(f"Status: {status[2]}, ObjVal: {status[0]:.5f}, MIPGap: {status[1]:.5f}\n")
                else:
                    logfile.write(f"Status: {status}\n")
                    
           
        #save all instances best dispersion values 
        save_objRes_xlsx('D',Solver_,GLM,dispersion, objs, filename = None)
        
        #close  log file
        logfile.close()
