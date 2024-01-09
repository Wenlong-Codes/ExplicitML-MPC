# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:24:02 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
# CSTR parameters
Ea = 5.0e4       # Activation energy [kJ/kmol]
R = 8.314    # Gas constant [J/mol/K]
k_0 = 8.46e6    # Arrhenius rate constant [m3/(kmol·hr)]
V = 1.0         # Volume [m3]
p_L = 1000.0   # Density [kg/m3]
Cp = 0.231     # Heat capacity [kJ/(kg·K)]
drH = -1.15e4    # Enthalpy of reaction [kJ/kmol]
F = 5.0        # Flowrate [m3/hr]   
C_A0s = 4       # Unstable steady state feed concentration [kmol/m3]
Q_s = 0.0       # Unstable steady state heat removing rate [kJ/hr]
T_0 = 300       # Inlet feed temperature [K]
C_As = 1.95
T_s = 402.0

def init_param():
    #===Parameter values of the continuous stirred tank reactor (CSTR)===#
    params = {
        'Ea'  : 5e4,       # Activation energy [kJ/kmol]
        'R'   : 8.314,     # Gas constant [J/mol/K]
        'k_0' : 8.46e6,    # Arrhenius rate constant [m3/(kmol·hr)]
        'V'   : 1,         # Volume [m3]
        'p_L' : 1000.0,    # Density [kg/m3]
        'Cp'  : 0.231,     # Heat capacity [kJ/(kg·K)]
        'drH' : -1.15e4,    # Enthalpy of reaction [kJ/kmol]
        'F'   : 5,         # Flowrate [m3/hr]   
    #===========================Model Varibles===========================#
        'C_A0': 4,         # Inlet feed concentration [kmol/m3]
        'T_0' : 300,       # Inlet feed temperature [K]
        'Q'   : 0,         # Heat removing rate [kJ/hr]
    #================ ========Simulation Parameters======================#
        't_end' : 0.01,    # Simulation Time [s]
        'sim_points' : 2,
    #==============Unstable Steady State Model Varibles==============#
        'C_As' : 1.95,     # Unstable steady state output concentration [kmol/m3]
        'T_s'  : 402,      # Unstable steady state feed temperature [K]      

    #=========================Control Parameters========================#
        'P': np.array([[1060, 22], 
                       [22, 0.52]]),
    #===============Unstable Steady State Control Varibles===============#
        'C_A0s' : 4 ,      # Unstable steady state feed concentration [kmol/m3]
        'Q_s' : 0.0}       # Unstable steady state heat removing rate [kJ/hr]
    return params

def FP_next_state(x_k, u_k):
    x_k1 = np.array([0.0, 0.0])
    
    dx1_dt = F/V*(-x_k[0]+(C_A0s-C_As)+u_k[0])-k_0*np.exp(-Ea/(R*(x_k[1]+T_s)))*(x_k[0]+C_As)**2
    dx2_dt = F/V*(-x_k[1]+(T_0-T_s))+(-drH)/(p_L*Cp)*k_0*np.exp(-Ea/(R*(x_k[1]+T_s)))*(x_k[0]+C_As)**2+(Q_s+u_k[1])/(p_L*Cp*V)

    dt = 0.01
    x_k1[0] = x_k[0] + dx1_dt*dt
    x_k1[1] = x_k[1] + dx2_dt*dt
    return x_k1[0], x_k1[1]


def mesh_points(C_As, T_s):
    # generating inputs and initial states for CSTR, all expressed in deviation form
    u1_list = np.linspace(-3.5, 3.5, 11, endpoint=True)
    u2_list = np.linspace(-5e5, 5e5, 51, endpoint=True)
    T_init = np.linspace(300, 600, 31, endpoint=True) - T_s 
    C_A_init = np.linspace(0, 6, 13, endpoint=True) - C_As
    return [u1_list, u2_list], T_init, C_A_init

def sieve_points(T_init, C_A_init, P):
    # sieve out initial states that lie outside of stability region
    T_st, C_A_st = [], []
    for T in T_init:
        for C_A in C_A_init:
            x = np.array([C_A, T])
            if x @ P @ x < 372:
                C_A_st.append(C_A)
                T_st.append(T)
    #print("number of initial conditions: {}".format(len(C_A_st))) 
    C_A_st = np.array(C_A_st)
    T_st = np.array(T_st)

    # every row is a pair of initial states within stability region
    x_devi = np.column_stack((C_A_st.T, T_st.T))  
    return C_A_st, T_st, x_devi

def plot_P_region(C_A_st, T_st):
    plt.figure(figsize=(12, 8))
    plt.scatter(C_A_st, T_st, s=10)
    plt.xlim((-2,2))
    plt.ylim((-100,100))

def generate_data(input_list):
    U_list, x_devi, params = input_list
    C_A_output, T_output = [], []
    C_A_input, T_input = [], []
    C_A0_input, Q_input = [], []
    cnt = 0
    total_length = len(U_list[0])*len(U_list[1])*len(x_devi)*len(U_list[0])
    with tqdm(total=total_length, file=sys.stdout) as pbar:
        for u1_k in U_list[0]:
            cnt += 1
            print(f'{cnt}\tstarting')
            for u2_k in U_list[1]:
                for (C_A_init, T_init) in x_devi:
                    x_k = np.array([C_A_init, T_init])
                    u_k = np.array([u1_k, u2_k])
                    x1_k1, x2_k1 = FP_next_state(x_k, u_k)
                    for u1_k1 in U_list[0]:
                        for u2_k1 in U_list[1]:
                            C_A_input.append(C_A_init)
                            T_input.append(T_init)
                            C_A0_input.append([u1_k, u1_k1])
                            Q_input.append([u2_k, u2_k1])
                            
                            x_k1 = np.array([x1_k1, x2_k1])
                            u_k1 = np.array([u1_k, u2_k])
                            x1_k2, x2_k2 = FP_next_state(x_k1, u_k1)
    
                            C_A_output.append([x1_k1, x1_k2])
                            T_output.append([x2_k1, x2_k2])         
                        pbar.update(1)
                    
    result = [C_A_input, T_input, C_A_output, T_output, C_A0_input, Q_input]
    print('one done!')
    return result

def save_data(C_A_input, T_input, C_A_output, T_output, C_A0_input, Q_input):
    data_dict = {'C_A_input'   :C_A_input,
                 'T_input'     :T_input,
                 'C_A_output'  :C_A_output,
                 'T_output'    :T_output,
                 'C_A0_input'  :C_A0_input,
                 'Q_input'     :Q_input}
    np.save('CSTR_open_loop_data_4D.npy', data_dict)

if __name__ == "__main__":   
    LOAD = 0
    if LOAD:
        sim_data = np.load('CSTR_open_loop_6D_FP.npy', allow_pickle=True).item()
    else:
        params = init_param()
        U_list, T_init, C_A_init = mesh_points(params['C_As'], params['T_s'])
        C_A_st, T_st, x_devi_total = sieve_points(T_init, C_A_init, params['P'])
        num = int(len(x_devi_total)/7)
        x_devi_list = [x_devi_total[i:i+num,:] for i in range(0, len(x_devi_total), num)]
        input_list = []
        for x_devi in x_devi_list:
            input_list.append([U_list, x_devi, params])
        pool = ProcessPoolExecutor(max_workers=len(input_list))
        print('Getting Explicit Solutions...')
        results = list(pool.map(generate_data, input_list))
        
        C_A_input, T_input, C_A_output, T_output, C_A0_input, Q_input = [], [], [], [], [], []
        for result in results:
            C_A_input.extend(result[0])
            T_input.extend(result[1])
            C_A_output.extend(result[2])
            T_output.extend(result[3])
            C_A0_input.extend(result[4])
            Q_input.extend(result[5])
            
        new_results = [C_A_input, T_input, C_A_output, T_output, C_A0_input, Q_input]
        np.save('FP_sim_6D.npy', new_results)












