# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:02:58 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")


class ML_Model(nn.Module):
    def __init__(self):
        super(ML_Model, self).__init__()
        self.W1 = nn.Linear(4, 12)
        self.W2 = nn.Linear(12, 24)
        self.W3 = nn.Linear(24, 8)
        self.W4 = nn.Linear(8, 2)
        
        self.W5 = nn.Linear(4, 12)
        self.W6 = nn.Linear(12, 24)
        self.W7 = nn.Linear(24, 8)
        self.W8 = nn.Linear(8, 2)
        self.RELU = nn.ReLU()
        self.tanh = nn.Tanh()
        
        
    def forward(self, x_k, u_k, u_k1):
        h1 = torch.cat((x_k, u_k), axis=2)
        h1 = self.tanh(self.W1(h1))
        h1 = self.tanh(self.W2(h1))
        h1 = self.tanh(self.W3(h1))
        x_k1_pred = self.tanh(self.W4(h1))
        
        h2 = torch.cat((x_k1_pred, u_k1), axis=2)
        h2 = self.tanh(self.W5(h2))
        h2 = self.tanh(self.W6(h2))
        h2 = self.tanh(self.W7(h2))
        x_k2_pred = self.tanh(self.W8(h2))
        
        return x_k1_pred, x_k2_pred


def generate_data(bds_x1, bds_x2, bds_u1, bds_u2):
    (L_x1, U_x1) = bds_x1
    (L_x2, U_x2) = bds_x2
    (L_u1, U_u1) = bds_u1
    (L_u2, U_u2) = bds_u2
    delta = round(U_x1 - L_x1,4)


    cnt_pts = 2
        
    delta = (U_x1-L_x1)/cnt_pts
    x1_step = np.linspace(L_x1, U_x1, cnt_pts, endpoint=False)
    x2_step = np.linspace(L_x2, U_x2, cnt_pts, endpoint=False)
    u1_step = np.linspace(L_u1, U_u1, cnt_pts, endpoint=False)
    u2_step = np.linspace(L_u2, U_u2, cnt_pts, endpoint=False)
    
    K_list, grid_pts_list = [], []
    for x1 in x1_step:
        for x2 in x2_step:
            for u1 in u1_step:
                for u2 in u2_step:
                    grid_pts_list.append([(x1, x2, u1, u2), (x1+delta, x2, u1, u2), (x1, x2+delta, u1, u2), (x1, x2, u1+delta, u2), (x1, x2, u1, u2+delta)])
    grid_pts_list = np.array(grid_pts_list)
    
    for grid_pts in grid_pts_list:
        input_xk = []
        input_uk = []
        input_uk1 = []
        for (x1, x2, u1, u2) in grid_pts:
            input_xk.append([x1, x2])
            input_uk.append([u1, u2])
            input_uk1.append([0, 0])
        input_xk = np.array(input_xk).reshape(-1, 1, 2)
        input_uk = np.array(input_uk).reshape(-1, 1, 2)
        input_uk1 = np.array(input_uk1).reshape(-1, 1, 2)

        input_xk = torch.from_numpy(input_xk).to(torch.float32)
        input_uk = torch.from_numpy(input_uk).to(torch.float32)
        input_uk1 = torch.from_numpy(input_uk1).to(torch.float32)
        x_k1, _ = model(input_xk, input_uk, input_uk1)
        value = x_k1.cpu().data.numpy()[:,0][:,1]
        W = np.column_stack((grid_pts, np.ones(grid_pts.shape[0]).reshape(-1,1)))
        W_inv = np.linalg.inv(W)
        K_list.append(np.dot(W_inv,value))

    return K_list, grid_pts_list


def grid_refine(awit_refine_grid_pts):
    delta = awit_refine_grid_pts[1][0]-awit_refine_grid_pts[0][0]
    ref_x1 = awit_refine_grid_pts[0][0]
    ref_x2 = awit_refine_grid_pts[0][1]
    ref_u1 = awit_refine_grid_pts[0][2]
    ref_u2 = awit_refine_grid_pts[0][3]
    refine_K_list, refine_grid_pts_list = generate_data((ref_x1, ref_x1+delta), (ref_x2, ref_x2+delta), (ref_u1, ref_u1+delta), (ref_u2, ref_u2+delta))
    return refine_K_list, refine_grid_pts_list


def screen_grid(K, grid_pts, model):
    delta = grid_pts[1][0]-grid_pts[0][0] 
    ref_x1 = grid_pts[0][0]
    ref_x2 = grid_pts[0][1]
    ref_u1 = grid_pts[0][2]
    ref_u2 = grid_pts[0][3]
    delta = round(delta, 4)
    
    if delta <= min_delta:
        return False
    
    pt_cnt = 3
    x1_step = np.linspace(ref_x1, ref_x1+delta, pt_cnt, endpoint=True)  #get x1 sampling points
    x2_step = np.linspace(ref_x2, ref_x2+delta, pt_cnt, endpoint=True)  #get x2 sampling points
    u1_step = np.linspace(ref_u1, ref_u1+delta, pt_cnt, endpoint=True)  #get u1 sampling points
    u2_step = np.linspace(ref_u2, ref_u2+delta, pt_cnt, endpoint=True)  #get u2 sampling points
    k, b = K[:-1], K[-1] #get coefficients of affine function
    
    true_value_list, est_value_list = [], []
    input_xk = []
    input_uk = []
    input_uk1 = []
    for x1 in x1_step:
        for x2 in x2_step:
            for u1 in u1_step:
                for u2 in u2_step:
    
                    input_xk.append([x1, x2])
                    input_uk.append([u1, u2])
                    input_uk1.append([0, 0])
                    est_value_list.append(np.array([x1, x2, u1, u2])@k+b)
    input_xk = np.array(input_xk).reshape(-1, 1, 2)
    input_uk = np.array(input_uk).reshape(-1, 1, 2)
    input_uk1 = np.array(input_uk1).reshape(-1, 1, 2)       
    
    
    input_xk = torch.from_numpy(input_xk).to(torch.float32)
    input_uk = torch.from_numpy(input_uk).to(torch.float32)
    input_uk1 = torch.from_numpy(input_uk1).to(torch.float32)
    
    x_k1, _ = model(input_xk, input_uk, input_uk1)
    true_value_list = x_k1.cpu().data.numpy()[:,0][:,1]
    
    est_value_list = np.array(est_value_list)
    err_list = abs(true_value_list-est_value_list)/(abs(true_value_list)+1e-9)*100
    ave_err = np.mean(err_list)
    # err_list = abs(true_value_list-est_value_list)
    # ave_err = np.linalg.norm(err_list)
    
    if ave_err >= 15:
        return True
    else:
        return False

def find_pt_area(pt_x1, pt_x2, pt_u1, pt_u2, overall_grid_pts_list):
    for i, grid_pts in enumerate(overall_grid_pts_list):
        delta = grid_pts[1][0]-grid_pts[0][0] 
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
            #print(f'x1:[{L_x:.4f}, {U_x:.4f}]\tx2:[{L_y:.4f}, {U_y:.4f}]\tu1:[{L_z:.4f}, {U_z:.4f}]')
            return i, [L_x1, U_x1, L_x2, U_x2, L_u1, U_u1, L_u2, U_u2]
    print('1 Not Found!')

def check_point1(pt_x1, pt_x2, pt_u1, pt_u2, overall_K_list, overall_grid_pts_list, model):
    area_id, bds = find_pt_area(pt_x1, pt_x2, pt_u1, pt_u2, overall_grid_pts_list)
    (coef_x1, coef_x2, coef_u1, coef_u2, coef_c) = overall_K_list[area_id]
    print(f'For point ({pt_x1}, {pt_x2}, {pt_u1}, {pt_u2}), it is located in the area of id:{area_id+1} (total: {len(overall_K_list)})')
    print(f'Corresponding affine function: ({coef_x1:.5}*x1)+({coef_x2:.5}*x2)+({coef_u1:.5}*u1)+({coef_u2:.5}*u2)+({coef_c:.5})')
    
    input_xk = np.array([pt_x1, pt_x2]).reshape(-1, 1, 2)
    input_uk = np.array([pt_u1, pt_u2]).reshape(-1, 1, 2)
    input_uk1 = np.array([0, 0]).reshape(-1, 1, 2)
    
    input_xk = torch.from_numpy(input_xk).to(torch.float32)
    input_uk = torch.from_numpy(input_uk).to(torch.float32)
    input_uk1 = torch.from_numpy(input_uk1).to(torch.float32)
    
    x_k1, _ = model(input_xk, input_uk, input_uk1)
    true_value = x_k1.cpu().data.numpy()[:,0][:,1][0]
    print(true_value)
    est_value = coef_x1*pt_x1 + coef_x2*pt_x2 + coef_u1*pt_u1 + coef_u2*pt_u2 + coef_c
    rel_err = abs(true_value-est_value)/(abs(true_value)+1e-9)*100
    print(f'true_value\t:{true_value}\nest_value\t:{est_value}\nrelative_err:{rel_err:.4f}%')
    return [coef_x1, coef_x2, coef_u1, coef_u2, coef_c], bds

def check_point1_new(pt_x1, pt_x2, pt_u1, pt_u2, overall_K_list, overall_grid_pts_list):
    area_id, bds = find_pt_area(pt_x1, pt_x2, pt_u1, pt_u2, overall_grid_pts_list)
    (coef_x1, coef_x2, coef_u1, coef_u2, coef_c) = overall_K_list[area_id]
    return [coef_x1, coef_x2, coef_u1, coef_u2, coef_c], bds

def discretization(K_list, grid_pts_list, model):
    overall_K_list, overall_grid_pts_list = [], []
    loop_cnt = 2
    while True:
        need_refine_K, need_refine_grid_pts = [], []
        for i in tqdm(range(len(grid_pts_list)),desc=f'Round {loop_cnt}: screening new area\t\t', file=sys.stdout):
            #print(i, state)
            if screen_grid(K_list[i], grid_pts_list[i], model):
                need_refine_grid_pts.append(grid_pts_list[i])
                need_refine_K.append(K_list[i])
            else:
                overall_grid_pts_list.append(grid_pts_list[i])
                overall_K_list.append(K_list[i])
                
        if len(need_refine_K) == 0 or len(need_refine_K) == 1:
            print(need_refine_K)
            tqdm.write(f'All area are identified! Total sub-area number: {len(overall_K_list)}')
            break
        else:
            K_list, grid_pts_list = [], []
            for need_refine_grid in tqdm(need_refine_grid_pts, desc=f'Round {loop_cnt}: discretizing known area', file=sys.stdout ):
                temp_k, temp_grid = grid_refine(need_refine_grid)
                K_list.extend(temp_k)
                grid_pts_list.extend(temp_grid)
        loop_cnt += 1
    return overall_K_list , overall_grid_pts_list

if __name__ == "__main__":
    model = torch.load('./Train_Model/FNN_CSTR_6D_FP.pkl').cpu()
    min_delta = round(2/(2*2*2), 5)
    LOAD = 0
    if LOAD:
        overall_K_list = np.load('./Grid_files/K_xk1_2.npy')
        overall_grid_pts_list = np.load('./Grid_files/Grid_xk1_2.npy')
    else:
        K_list, grid_pts_list = generate_data((-1, 1), (-1, 1), (-1, 1), (-1, 1))
        overall_K_list, overall_grid_pts_list = discretization(K_list, grid_pts_list, model)
        np.save('./Grid_files/Grid_xk1_2.npy', overall_grid_pts_list)
        np.save('./Grid_files/K_xk1_2.npy', overall_K_list)
        
    pt_x1, pt_x2, pt_u1, pt_u2 = 0.1, -0.2, 0.2, 0.2
    coef_list, bds= check_point1(pt_x1, pt_x2, pt_u1, pt_u2, overall_K_list, overall_grid_pts_list, model)
    





