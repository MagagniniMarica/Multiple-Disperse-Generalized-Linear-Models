# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini

This is the main file to solve a set of instances of the (A-PDP-GLM) problem, i.e.
 maximum accuracy problem. 
"""


from Funs import Dataset_selection, S_Q_, params_, PCD_problem_MAXACC_, save_results_coeff_xlsx, save_objRes_xlsx


# 
# GML : Linear Regression (Lin), Logistic Regression (Log), Poisson Regression (Poi)
#
GLM_list = ['Lin', 'Log', 'Poi']#'Lin', 'Log', 'Poi'

#
# Dispersion Notion: 'l1', 'l2', 'dsa', 'o1'  
# 
Dispersion_list = ['l1', 'l2', 'dsa', 'o1' ]


#
# Solver optimal or local
#
Solvers = ['gurobi_persistent', 'ipopt']

#
# P-CD_GLM Parameters
#
P = 3                           # Number of new GLMs
M = 100                         # big M



# Dispersion percentage gamma = g_pc*gamma_max  (or gamma = gamma_max - g_pc) with g_pc in GAMMA_percentage
GAMMA_percentage_disp = {'dsa': [1,2,3], '-':[.85, .75, .65]}

#%% Function to retrieve a feasible starting solution, if there is, from the previous computations.
def feas_sol(tau, theta, g_pc, gamma, dispersion, pool):
    """
    

    Parameters
    ----------
    tau : float 
        accuracy threshold
    theta : int
        sparcity in [6,7,8,9]
    g_pc : float
        dispersion level.
    pool : list of tuples
        pool[(tau,theta, g_pc)] = {
            "gamma" : gamma
            "obj" : obj,
            "SP" : SP}

    Returns
    -------
    If available it returns a feasible solution for the instance of the MAXACC problem
    to be solved
    
    SP_feas : list of models
    obj_feas : float

    """
    def build_g_pc_feas(dispersion, g_pc):
        if dispersion == 'dsa':
            g_pc_feas = g_pc - 1 
        else:
            g_pc_feas = round(g_pc + 0.1,2)
        return g_pc_feas
        
    # Feasible solution, already known, for the specific instance of the MAX ACC
    # problem. No feasible solution is known for the first case, 
    #i.e. theta = 6, g_pc = 0.85 or 1
    SP_feas, obj_feas = ([], 1e10)
    
    if theta == 6 and (g_pc == 0.85 or g_pc == 1):
        pass
    elif theta == 6 and (g_pc != 0.85 and g_pc != 1):
        g_pc_feas = build_g_pc_feas(dispersion, g_pc)
        try:
            SP_feas = pool[(tau, theta, g_pc_feas)]['SP']
            obj_feas = pool[(tau, theta, g_pc_feas)]['obj']
        except:
            print('No feasible solution available.')
    elif theta > 6 and (g_pc == 0.85 or g_pc == 1):
         # gamma(theta) <= gamma(theta - 1)
        try:
            if gamma <= pool[(tau, theta-1, g_pc)]['gamma']:
                SP_feas = pool[(tau, theta-1, g_pc)]['SP']
                obj_feas = pool[(tau, theta-1, g_pc)]['obj']
        except KeyError:
            print("No feasible solution available.")

    else: # theta > 6 and g_pc > 0.85/1
        # if there exists a solution that satisfies the current dispersion,
        #choose the one with the min err among all
        for ii in pool:
            if ii[0] == tau and pool[ii]['gamma']>= gamma and pool[ii]['obj'] <= obj_feas:
                obj_feas = pool[ii]['obj']
                SP_feas = pool[ii]['SP']
        
        if SP_feas == []:
            print('No feasible solution available.')
    
    
    return SP_feas, obj_feas


#%% Executions
#
# Multi-start approach for the heuristic
#
runs_vns = 10


for GLM, Solver_ in [ ('Lin','gurobi_persistent'), ('Lin','ipopt'),('Log','ipopt'),('Poi','ipopt')]:
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
    SQ, SQ_obj = S_Q_(GLM, features)  
    
    
    TimeLimit_vns = {'Lin': 120, 'Log': 120, 'Poi': 300}.get(GLM, None)     # Timelimit single vns single run
        
    for dispersion in Dispersion_list:

        if Solver_ == 'gurobi_persistent':
            logfile = open(f"A_log_solver_status_{GLM}_{dispersion}.txt", "a")
        else:
            logfile = open(f"A_heur_solver_status_{GLM}_{dispersion}.txt", "a")
        
        print('---------------------------------------------------------------------------')
        print(f'GLM: {GLM}, dispersion: {dispersion}. N = {N}, J = {J}')
        print('Parameters:')
        #
        # Parameters
        #
        parameters = params_(GLM, dispersion)

        #Indexed by t_pc, theta, g_pc, contains gamma, SP_obj and SP
        pool = {}
        
       
        objs = []
        for par in parameters.values: 
            # par = parameters.values[1]
            tau, theta, gamma_max =( par.item(0), int(par[1]), par.item(2))

            #
            # Define gamma iterations
            #
            if gamma_max == 0  :
                GAMMA_percentage = []
            elif dispersion == 'dsa':
                GAMMA_percentage = GAMMA_percentage_disp['dsa']
            else:
                GAMMA_percentage = GAMMA_percentage_disp['-']
            
            
            for g_pc  in GAMMA_percentage:
                if dispersion == 'dsa':
                    gamma = int(gamma_max - g_pc)
                    gamma = max(0, gamma)
                    print(f' tau :{tau}, theta: {theta},  gamma: {gamma} (gamma_max {gamma_max})')
            
                else:
                    gamma  = round(gamma_max * g_pc,3)
                    print(f' tau :{tau}, theta: {theta},  gamma: {gamma} ({g_pc*100}% gamma_max ({round(gamma_max,3)}))')
                
                # Feasible solution 
                SP_feas, obj_feas = feas_sol(tau, theta, g_pc, gamma, dispersion, pool)
                
                #
                # Execution
                #   
                SP, SP_obj, status = PCD_problem_MAXACC_(GLM, Solver_, 
                                          dataset,target,features, 
                                          
                                          SQ, P, gamma, tau, theta, M, X0,
                                          dispersion,  runs_vns,TimeLimit_vns,
                                          SP_feas, obj_feas)
                
                if SP != []:
                    # Save Results
                    save_results_coeff_xlsx('A',Solver_, GLM, dispersion, theta,tau, gamma, 
                                                SP,SP_obj.item(), SQ, SQ_obj,  features, filename = None)
                
                objs.append((tau, theta, gamma,  SP_obj))
                
                # Record all gamma, SP and SP_obj associated to tau, theta and g_pc 
                pool[(tau,theta, g_pc)] = {
                    "gamma" : gamma,
                    "obj" : SP_obj,
                    "SP" : SP}
                
                #
                #  Write log -- Solver/heuristic status
                #
                logfile.write(f"GLM={GLM}, Dispersion={dispersion}, tau={tau:.5f}, theta={theta}, gamma={gamma} ")
                if Solver_ == 'gurobi_persistent':
                    logfile.write(f"Status: {status[2]}, ObjVal: {status[0]:.5f}, MIPGap: {status[1]:.5f}\n")
                else:
                    logfile.write(f'Status: {status}\n')
           
            
        save_objRes_xlsx('A',Solver_,GLM,dispersion, objs, filename = None)
        
        #close gurobi log file
        logfile.close()
