# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:15:46 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
from tqdm import tqdm

import time
from concurrent.futures import ProcessPoolExecutor
from sympy import symbols, diff, Matrix
import sys


def g(x1, x2, u1, u2, P_u2, P_x2, P_xu, P_u1, P_x1, P_c):
    x = Matrix([[x1],
                [x2]])
    u = Matrix([[u1],
                [u2]])
    z = u.T*P_u2*u+ x.T*P_x2*x + x.T*P_xu*u + P_u1*u + P_x1*x + P_c
    return z


def Solve_MPQP(M, N, P, A, B, C, D, E, F, G, bds_x1, bds_x2, bds_u1, bds_u2, bds_u1_k1, bds_u2_k1):
    L_x1, U_x1 = bds_x1
    L_x2, U_x2 = bds_x2
    L_u1, U_u1 = bds_u1
    L_u2, U_u2 = bds_u2
    L_u1_k1, U_u1_k1 = bds_u1_k1
    L_u2_k1, U_u2_k1 = bds_u2_k1
    
    M1_1 = E.T@M@E + B.T@M@B + N
    M1_2 = E.T@M@F
    M1_3 = F.T@M@E
    M1_4 = F.T@M@F + N
    M1_12 = np.column_stack((M1_1, M1_2))
    M1_34 = np.column_stack((M1_3, M1_4))
    M1 = np.row_stack((M1_12, M1_34))
    
    M2 = D.T@M@D + A.T@M@A
    
    M3_1 = 2*(D.T@M@E + A.T@M@B)
    M3_2 = 2*D.T@M@F
    M3 = np.column_stack((M3_1, M3_2))
    
    M4_1 = 2*(G.T@M@E + C.T@M@B)
    M4_2 = 2*G.T@M@F
    M4 = np.column_stack((M4_1, M4_2))
    
    M5 = 2*(G.T@M@D + C.T@M@A)
    M6 = G.T@M@G + C.T@M@C
    
    Q = 2*M1
    Q_t = M2
    H_t = M3
    c = M4
    c_t = M5
    c_c = M6
    
    P_u2 = Matrix(B.T@P@B)
    P_x2 = Matrix(A.T@P@A-0.999*P)
    P_xu = Matrix(2*A.T@P@B)
    P_u1 = Matrix(2*C.T@P@B)
    P_x1 = Matrix(2*C.T@P@A)
    P_c = Matrix(C.T@P@C)
    
    mid_x1 = (L_x1+U_x1)/2
    mid_x2 = (L_x2+U_x2)/2
    mid_u1 = (L_u1+U_u1)/2
    mid_u2 = (L_u2+U_u2)/2

    x1, x2, u1, u2 = symbols('x1 x2 u1 u2')
    func_g = g(x1, x2, u1, u2, P_u2, P_x2, P_xu, P_u1, P_x1, P_c)
    func_d_x1 = diff(func_g, x1)
    func_d_x2 = diff(func_g, x2)
    func_d_u1 = diff(func_g, u1)
    func_d_u2 = diff(func_g, u2)
    
    d_x1_value = func_d_x1.evalf(subs={x1: mid_x1, x2: mid_x2, u1:mid_u1, u2:mid_u2}).tolist()[0][0]
    d_x2_value = func_d_x2.evalf(subs={x1: mid_x1, x2: mid_x2, u1:mid_u1, u2:mid_u2}).tolist()[0][0]
    d_u1_value = func_d_u1.evalf(subs={x1: mid_x1, x2: mid_x2, u1:mid_u1, u2:mid_u2}).tolist()[0][0]
    d_u2_value = func_d_u2.evalf(subs={x1: mid_x1, x2: mid_x2, u1:mid_u1, u2:mid_u2}).tolist()[0][0]
    g_value = func_g.evalf(subs={x1: mid_x1, x2: mid_x2, u1:mid_u1, u2:mid_u2}).tolist()[0][0]
    
    A_ = np.array([[1, 0, 0, 0],
                   [-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, -1],
                   [d_u1_value, d_u2_value, 0, 0]])
    
    temp1 = np.array([d_x1_value, d_x2_value, d_u1_value, d_u2_value]).reshape(1,4)
    temp2 = np.array([mid_x1, mid_x2, mid_u1, mid_u2]).reshape(4,1)
    temp = temp1@temp2
    
    b = np.array([[U_u1],
                  [-L_u1],
                  [U_u2],
                  [-L_u2],
                  [U_u1_k1],
                  [-L_u1_k1],
                  [U_u2_k1],
                  [-L_u2_k1],
                  [temp[0][0]-g_value]])
                  
    F_ = np.array([[0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [-d_x1_value, -d_x2_value]])

    
    A_t = np.array([[1, 0],
                    [-1,0],
                    [0, 1],
                    [0,-1]])
    
    b_t = np.array([[U_x1],
                    [-L_x1],
                    [U_x2],
                    [-L_x2]])
    
    prog = MPQP_Program(A=A_, b=b, c=c.T, H=H_t.T, Q=Q, A_t=A_t, b_t=b_t, F=F_, c_c=c_c, c_t=c_t.T, Q_t=Q_t)
    prog.process_constraints()
    sol = solve_mpqp(prog, mpqp_algorithm.combinatorial)
    return sol

def LoadData():
    grid_xk1_list1 = np.load('./Grid_files/Grid_xk1_1.npy')
    K_xk1_list1 = np.load('./Grid_files/K_xk1_1.npy')
    grid_xk1_list2 = np.load('./Grid_files/Grid_xk1_2.npy')
    K_xk1_list2 = np.load('./Grid_files/K_xk1_2.npy')
    
    grid_xk2_list1 = np.load('./Grid_files/Grid_xk2_1.npy')
    K_xk2_list1 = np.load('./Grid_files/K_xk2_1.npy')
    grid_xk2_list2 = np.load('./Grid_files/Grid_xk2_2.npy')
    K_xk2_list2 = np.load('./Grid_files/K_xk2_2.npy')
    
    all_grid_list = [grid_xk1_list1, grid_xk1_list2, grid_xk2_list1, grid_xk2_list2]
    all_K_list = [K_xk1_list1, K_xk1_list2, K_xk2_list1, K_xk2_list2]
    return all_grid_list, all_K_list

def FindBoundary(all_grid_list, axis_name):
    if axis_name == 'u1_k1' or axis_name == 'u2_k1':
        all_grid_list = all_grid_list[2:]
        
    bds_list = []
    total_bds_raw_list = []
    for i, grid_pts_list in enumerate(all_grid_list):
        bds_raw_list = []
        for grid_pts in grid_pts_list:
            if axis_name == 'x1':
                bds_left = grid_pts[0][0]
                bds_right = grid_pts[1][0]
            elif axis_name == 'x2':
                bds_left = grid_pts[0][1]
                bds_right = grid_pts[2][1]
            elif axis_name == 'u1':
                bds_left = grid_pts[0][2]
                bds_right = grid_pts[3][2]
            elif axis_name == 'u2':
                bds_left = grid_pts[0][3]
                bds_right = grid_pts[4][3]
            elif axis_name == 'u1_k1':
                bds_left = grid_pts[0][4]
                bds_right = grid_pts[5][4]
            elif axis_name == 'u2_k1':
                bds_left = grid_pts[0][5]
                bds_right = grid_pts[6][5]
            bds_raw_list.append(round(bds_left,6))
            bds_raw_list.append(round(bds_right,6))
        bds_raw_list = set(bds_raw_list)
        bds_raw_list_list = list(bds_raw_list)
        bds_raw_list_list.sort()
        total_bds_raw_list.append(bds_raw_list_list)

    for bds_raw_list_list in total_bds_raw_list:
        for bds in bds_raw_list_list:
            if bds not in bds_list:
                bds_list.append(bds)

    bds_list = list(set(bds_list))
    bds_list.sort()
    return bds_list

def check_point_k1(pt_x1, pt_x2, pt_u1, pt_u2, K_list, grid_pts_list):
    for i, grid_pts in enumerate(grid_pts_list):
        delta = round(grid_pts[1][0]-grid_pts[0][0], 5)
        L_x1 = grid_pts[0][0] #get reference x1 point of the grid
        L_x2 = grid_pts[0][1] #get reference x2 point of the grid
        L_u1 = grid_pts[0][2] #get reference u1 point of the grid
        L_u2 = grid_pts[0][3] #get reference u2 point of the grid
        U_x1 = L_x1+delta
        U_x2 = L_x2+delta
        U_u1 = L_u1+delta
        U_u2 = L_u2+delta
        arr_bds = np.array([U_x1, -L_x1, U_x2, -L_x2, U_u1, -L_u1, U_u2, -L_u2]).reshape(-1,1)
        arr_coef = np.array([[1, 0, 0, 0],
                             [-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, -1]])
        arr_pt = np.array([pt_x1, pt_x2, pt_u1, pt_u2]).reshape(-1, 1)
        if all(arr_coef@arr_pt<=arr_bds):
            (coef_x1, coef_x2, coef_u1, coef_u2, coef_c) = K_list[i]
            return [coef_x1, coef_x2, coef_u1, coef_u2, coef_c]
    print(pt_x1, pt_x2, pt_u1, pt_u2)
    raise ValueError('k1 Not Found!')
    
def check_point_k2(pt_x1, pt_x2, pt_u1, pt_u2, pt_u1_k1, pt_u2_k1, K_list, grid_pts_list):
    for i, grid_pts in enumerate(grid_pts_list):
        delta = round(grid_pts[1][0]-grid_pts[0][0], 5)
        L_x1 = grid_pts[0][0] #get reference x1 point of the grid
        L_x2 = grid_pts[0][1] #get reference x2 point of the grid
        L_u1 = grid_pts[0][2] #get reference u1 point of the grid
        L_u2 = grid_pts[0][3] #get reference u2 point of the grid
        L_u1_k1 = grid_pts[0][4] #get reference u1_k1 point of the grid
        L_u2_k1 = grid_pts[0][5] #get reference u2_k1 point of the grid
        
        U_x1 = L_x1+delta
        U_x2 = L_x2+delta
        U_u1 = L_u1+delta
        U_u2 = L_u2+delta
        U_u1_k1 = L_u1_k1+delta
        U_u2_k1 = L_u2_k1+delta
        
        arr_bds = np.array([U_x1, -L_x1, U_x2, -L_x2, U_u1, -L_u1, U_u2, -L_u2, U_u1_k1, -L_u1_k1, U_u2_k1, -L_u2_k1]).reshape(-1,1)
        arr_coef = np.array([[1, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, -1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, -1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, -1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, -1, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, -1]])
        arr_pt = np.array([pt_x1, pt_x2, pt_u1, pt_u2, pt_u1_k1, pt_u2_k1]).reshape(-1, 1)
        if all(arr_coef@arr_pt<=arr_bds):
            (coef_x1, coef_x2, coef_u1, coef_u2, coef_u1_k1, coef_u2_k1, coef_c) = K_list[i]
            return [coef_x1, coef_x2, coef_u1, coef_u2, coef_u1_k1, coef_u2_k1, coef_c]
    print(pt_x1, pt_x2, pt_u1, pt_u2, pt_u1_k1, pt_u2_k1)
    raise ValueError('k2 Not Found!')
    


def get_scaler_info(input_scaler, output_scaler):
    x1_max, x2_max, u1_max, u2_max, u1_k1_max, u2_k1_max = input_scaler.data_max_
    x1_min, x2_min, u1_min, u2_min, u1_k1_min, u2_k1_min = input_scaler.data_min_
    
    x1_k1_max, x2_k1_max, x1_k2_max, x2_k2_max = output_scaler.data_max_
    x1_k1_min, x2_k1_min, x1_k2_min, x2_k2_min = output_scaler.data_min_
    
    x1_sum = x1_max + x1_min #x1
    x1_diff = x1_max - x1_min
    
    x2_sum = x2_max + x2_min #x2
    x2_diff = x2_max - x2_min
    
    u1_sum = u1_max + u1_min #u1
    u1_diff = u1_max - u1_min
    
    u2_sum = u2_max + u2_min #u2
    u2_diff = u2_max - u2_min
    
    u1_k1_sum = u1_k1_max + u1_k1_min #u1_k1
    u1_k1_diff = u1_k1_max - u1_k1_min
    
    u2_k1_sum = u2_k1_max + u2_k1_min #u2_k1
    u2_k1_diff = u2_k1_max - u2_k1_min
    
    x1_k1_sum = x1_k1_max + x1_k1_min #x1_k1
    x1_k1_diff = x1_k1_max - x1_k1_min
    
    x2_k1_sum = x2_k1_max + x2_k1_min #x2_k1
    x2_k1_diff = x2_k1_max - x2_k1_min
    
    x1_k2_sum = x1_k2_max + x1_k2_min #x1_k2
    x1_k2_diff = x1_k2_max - x1_k2_min
    
    x2_k2_sum = x2_k2_max + x2_k2_min #x2_k2
    x2_k2_diff = x2_k2_max - x2_k2_min
    
    sum_list = [x1_sum, x2_sum, u1_sum, u2_sum, u1_k1_sum, u2_k1_sum, x1_k1_sum, x2_k1_sum, x1_k2_sum, x2_k2_sum]
    diff_list = [x1_diff, x2_diff, u1_diff, u2_diff, u1_k1_diff, u2_k1_diff, x1_k1_diff, x2_k1_diff, x1_k2_diff, x2_k2_diff]
    return sum_list, diff_list

def coef_rescale_k1(coef_list1, coef_list2, sum_list, diff_list):
    x1_sum, x2_sum, u1_sum, u2_sum, _, _, x1_k1_sum, x2_k1_sum, _, _ = sum_list
    x1_diff, x2_diff, u1_diff, u2_diff, _, _, x1_k1_diff, x2_k1_diff, _, _ = diff_list

    A1_, B1_, C1_, D1_, E1_ = coef_list1
    A1 = A1_*x1_k1_diff/x1_diff
    B1 = B1_*x1_k1_diff/x2_diff
    C1 = C1_*x1_k1_diff/u1_diff
    D1 = D1_*x1_k1_diff/u2_diff
    E1 = (E1_+x1_k1_sum/x1_k1_diff-A1_*x1_sum/x1_diff-B1_*x2_sum/x2_diff-C1_*u1_sum/u1_diff-D1_*u2_sum/u2_diff)*x1_k1_diff/2
    
    A2_, B2_, C2_, D2_, E2_ = coef_list2
    A2 = A2_*x2_k1_diff/x1_diff
    B2 = B2_*x2_k1_diff/x2_diff
    C2 = C2_*x2_k1_diff/u1_diff
    D2 = D2_*x2_k1_diff/u2_diff
    E2 = (E2_+x2_k1_sum/x2_k1_diff-A2_*x1_sum/x1_diff-B2_*x2_sum/x2_diff-C2_*u1_sum/u1_diff-D2_*u2_sum/u2_diff)*x2_k1_diff/2   
    return [A1, B1, C1, D1, E1],  [A2, B2, C2, D2, E2]  

def coef_rescale_k2(coef_list1, coef_list2, sum_list, diff_list):
    x1_sum, x2_sum, u1_sum, u2_sum, u1_k1_sum, u2_k1_sum, x1_k1_sum, x2_k1_sum, x1_k2_sum, x2_k2_sum = sum_list
    x1_diff, x2_diff, u1_diff, u2_diff, u1_k1_diff, u2_k1_diff, x1_k1_diff, x2_k1_diff, x1_k2_diff, x2_k2_diff = diff_list

    A1_, B1_, C1_, D1_, E1_, F1_, G1_ = coef_list1
    A1 = A1_*x1_k2_diff/x1_diff
    B1 = B1_*x1_k2_diff/x2_diff
    C1 = C1_*x1_k2_diff/u1_diff
    D1 = D1_*x1_k2_diff/u2_diff
    E1 = E1_*x1_k2_diff/u1_k1_diff
    F1 = F1_*x1_k2_diff/u2_k1_diff
    G1 = (G1_+x1_k2_sum/x1_k2_diff-A1_*x1_sum/x1_diff-B1_*x2_sum/x2_diff-C1_*u1_sum/u1_diff-D1_*u2_sum/u2_diff-E1_*u1_k1_sum/u1_k1_diff-F1_*u2_k1_sum/u2_k1_diff)*x1_k2_diff/2
    
    A2_, B2_, C2_, D2_, E2_, F2_, G2_ = coef_list2
    A2 = A2_*x2_k2_diff/x1_diff
    B2 = B2_*x2_k2_diff/x2_diff
    C2 = C2_*x2_k2_diff/u1_diff
    D2 = D2_*x2_k2_diff/u2_diff
    E2 = E2_*x2_k2_diff/u1_k1_diff
    F2 = F2_*x2_k2_diff/u2_k1_diff
    G2 = (G2_+x2_k2_sum/x2_k2_diff-A2_*x1_sum/x1_diff-B2_*x2_sum/x2_diff-C2_*u1_sum/u1_diff-D2_*u2_sum/u2_diff-E2_*u1_k1_sum/u1_k1_diff-F2_*u2_k1_sum/u2_k1_diff)*x2_k2_diff/2
    return [A1, B1, C1, D1, E1, F1, G1],  [A2, B2, C2, D2, E2, F2, G2]
    

def GetExplicitSol(input_list):
    (M, N, P, all_grid_list, all_K_list, input_scaler, output_scaler, bds_x1_pair_sublist, bds_x2_pair_sublist, all_U_bds_pair_list) = input_list
    
    grid_xk1_list1, grid_xk1_list2, grid_xk2_list1, grid_xk2_list2 = all_grid_list
    K_xk1_list1, K_xk1_list2, K_xk2_list1, K_xk2_list2 = all_K_list
    
    bds_u1_pair_list, bds_u2_pair_list, bds_u1_k1_pair_list, bds_u2_k1_pair_list = all_U_bds_pair_list
    sum_list, diff_list = get_scaler_info(input_scaler, output_scaler)
    x_space_sol_list = []
    
    total_length = len(bds_x1_pair_sublist)*len(bds_x2_pair_sublist)*len(bds_u1_pair_list)*len(bds_u2_pair_list)
    with tqdm(total=total_length, file=sys.stdout) as pbar:
        for x1_pair_ in bds_x1_pair_sublist:
            for x2_pair_ in bds_x2_pair_sublist:
                ExplicitMPC_u_list = []
                for u1_pair_ in bds_u1_pair_list:
                    for u2_pair_ in bds_u2_pair_list:
                        for u1_k1_pair_ in bds_u1_k1_pair_list:
                            for u2_k1_pair_ in bds_u2_k1_pair_list:
                                pt_x1_ = x1_pair_[0] + 1e-5
                                pt_x2_ = x2_pair_[0] + 1e-5
                                pt_u1_ = u1_pair_[0] + 1e-5
                                pt_u2_ = u2_pair_[0] + 1e-5
                                pt_u1_k1_ = u1_k1_pair_[0] + 1e-5
                                pt_u2_k1_ = u2_k1_pair_[0] + 1e-5
                                
                                coef_list_k1_1_ = check_point_k1(pt_x1_, pt_x2_, pt_u1_, pt_u2_, K_xk1_list1, grid_xk1_list1)
                                coef_list_k1_2_ = check_point_k1(pt_x1_, pt_x2_, pt_u1_, pt_u2_, K_xk1_list2, grid_xk1_list2)
                                
                                coef_list_k2_1_ = check_point_k2(pt_x1_, pt_x2_, pt_u1_, pt_u2_, pt_u1_k1_, pt_u2_k1_, K_xk2_list1, grid_xk2_list1)
                                coef_list_k2_2_ = check_point_k2(pt_x1_, pt_x2_, pt_u1_, pt_u2_, pt_u1_k1_, pt_u2_k1_, K_xk2_list2, grid_xk2_list2)
                                
                                coef_list_k1_1, coef_list_k1_2 = coef_rescale_k1(coef_list_k1_1_, coef_list_k1_2_, sum_list, diff_list)
                                coef_list_k2_1, coef_list_k2_2 = coef_rescale_k2(coef_list_k2_1_, coef_list_k2_2_, sum_list, diff_list)
                                                   
                                A = np.array([[coef_list_k1_1[0], coef_list_k1_1[1]],
                                              [coef_list_k1_2[0], coef_list_k1_2[1]]])
                                
                                B = np.array([[coef_list_k1_1[2], coef_list_k1_1[3]],
                                              [coef_list_k1_2[2], coef_list_k1_2[3]]])
                                
                                C = np.array([[coef_list_k1_1[4]],
                                              [coef_list_k1_2[4]]])
                                
                                D = np.array([[coef_list_k2_1[0], coef_list_k2_1[1]],
                                              [coef_list_k2_2[0], coef_list_k2_2[1]]])
                                
                                E = np.array([[coef_list_k2_1[2], coef_list_k2_1[3]],
                                              [coef_list_k2_2[2], coef_list_k2_2[3]]])
                                
                                F = np.array([[coef_list_k2_1[4], coef_list_k2_1[5]],
                                              [coef_list_k2_2[4], coef_list_k2_2[5]]])
                                
                                G = np.array([[coef_list_k2_1[6]],
                                              [coef_list_k2_2[6]]])
                            
                                x1_pair = [0.]*2
                                x2_pair = [0.]*2
                                u1_pair = [0.]*2
                                u2_pair = [0.]*2
                                u1_k1_pair = [0.]*2
                                u2_k1_pair = [0.]*2
                                
                                a = [x1_pair_[0], x2_pair_[0], u1_pair_[0], u2_pair_[0], u1_k1_pair_[0], u2_k1_pair_[0]]
                                b = [x1_pair_[1], x2_pair_[1], u1_pair_[1], u2_pair_[1], u1_k1_pair_[1], u2_k1_pair_[1]]
                                x1_pair[0], x2_pair[0], u1_pair[0], u2_pair[0], u1_k1_pair[0], u2_k1_pair[0] = input_scaler.inverse_transform([a])[0]
                                x1_pair[1], x2_pair[1], u1_pair[1], u2_pair[1], u1_k1_pair[1], u2_k1_pair[1] = input_scaler.inverse_transform([b])[0]
    
                                sol = Solve_MPQP(M, N, P, A, B, C, D, E, F, G, 
                                          x1_pair, x2_pair, 
                                          u1_pair, u2_pair,
                                          u1_k1_pair, u2_k1_pair)
    
                                if len(sol.critical_regions) != 0:
                                    #print('Not Empty!')
                                    ExplicitMPC_u_list.append(sol)     
                                # print(sol)
                                # return sol #test
                        pbar.update(1)             
                if len(ExplicitMPC_u_list) != 0:
                    x_bds = [x1_pair[1], -x1_pair[0], x2_pair[1], -x2_pair[0]]
                    x_space_sol_list.append([x_bds, ExplicitMPC_u_list])
    print('one subprocess finished!')
    return x_space_sol_list


if __name__ == "__main__":
    all_grid_list, all_K_list = LoadData()
    
    bds_x1_list = FindBoundary(all_grid_list, 'x1')
    bds_x2_list = FindBoundary(all_grid_list, 'x2')
    bds_u1_list = FindBoundary(all_grid_list, 'u1')
    bds_u2_list = FindBoundary(all_grid_list, 'u2')
    bds_u1_k1_list = FindBoundary(all_grid_list, 'u1_k1')
    bds_u2_k1_list = FindBoundary(all_grid_list, 'u2_k1')
    
    bds_x1_pair_list = [[bds_x1_list[i], bds_x1_list[i+1]] for i in range(len(bds_x1_list)-1)]
    bds_x2_pair_list = [[bds_x2_list[i], bds_x2_list[i+1]] for i in range(len(bds_x2_list)-1)]
    bds_u1_pair_list = [[bds_u1_list[i], bds_u1_list[i+1]] for i in range(len(bds_u1_list)-1)]
    bds_u2_pair_list = [[bds_u2_list[i], bds_u2_list[i+1]] for i in range(len(bds_u2_list)-1)]
    bds_u1_k1_pair_list = [[bds_u1_k1_list[i], bds_u1_k1_list[i+1]] for i in range(len(bds_u1_k1_list)-1)]
    bds_u2_k1_pair_list = [[bds_u2_k1_list[i], bds_u2_k1_list[i+1]] for i in range(len(bds_u2_k1_list)-1)]
    
    all_U_bds_pair_list = [bds_u1_pair_list, bds_u2_pair_list, bds_u1_k1_pair_list, bds_u2_k1_pair_list]

    
    input_scaler = np.load('./Train_Model/input_scaler_FP.npy', allow_pickle=True).item()
    output_scaler = np.load('./Train_Model/output_scaler_FP.npy', allow_pickle=True).item()
    
    
    M = np.array([[500.0, 0.0],
                    [0.0, 0.5]])
    
    N = np.array([[1.0, 0.0],
                  [0.0, 8.0e-11]])
    
    P = np.array([[1060.0, 22.0],
                    [22.0, 0.52]])
    
    
    bds_x1_pair_each_cnt = int(len(bds_x1_pair_list)/4)
    bds_x2_pair_each_cnt = int(len(bds_x2_pair_list)/4)
    bds_x1_pair_multi_threading_list = [bds_x1_pair_list[i:i+bds_x1_pair_each_cnt] for i in range(0, len(bds_x1_pair_list), bds_x1_pair_each_cnt)]
    bds_x2_pair_multi_threading_list = [bds_x2_pair_list[i:i+bds_x2_pair_each_cnt] for i in range(0, len(bds_x2_pair_list), bds_x2_pair_each_cnt)]
    worker = int(len(bds_x1_pair_multi_threading_list)*len(bds_x1_pair_multi_threading_list))
    
    # del bds_x1_list, bds_x2_list, bds_u1_list, bds_u2_list, bds_u1_k1_list, bds_u2_k1_list
    # del bds_x1_pair_list, bds_x2_pair_list, bds_u1_pair_list, bds_u2_pair_list, bds_u1_k1_pair_list, bds_u2_k1_pair_list
    
    
    input_list = []
    for bds_x1_pair_sublist in bds_x1_pair_multi_threading_list:
        for bds_x2_pair_sublist in bds_x2_pair_multi_threading_list:
            input_list.append([M, N, P, all_grid_list, all_K_list, input_scaler, output_scaler, bds_x1_pair_sublist, bds_x2_pair_sublist, all_U_bds_pair_list])

    
    pool = ProcessPoolExecutor(max_workers=worker)
    print('Getting Explicit MPC Solutions...')
    time1 = time.time()
    
    #solving these mpQPs in a paralle mode to save time
    results = list(pool.map(GetExplicitSol, input_list))
    time2 = time.time()
    print(time2-time1)
    
    new_results = []
    for re in results:
        if len(re) != 0:
            new_results.append(re)
    ExplicitMPC_Sols = []
    for parts in new_results:
        for item in parts:
            ExplicitMPC_Sols.append(item)
            
    #save the solutions
    np.save('ExplicitML-MPC_Sols.npy', ExplicitMPC_Sols)
    print('All Done!')
    
    
    
    
    
    
    
    
    
    
    
    
