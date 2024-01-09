# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:49:50 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")

w_x = np.array([[1, 0],
                [-1, 0],
                [0, 1],
                [0, -1]])

# MPC parameters
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

# Weights of Objective function
P = np.array([[1060.0, 22.0], 
               [22.0, 0.52]])
Q1 = np.array([[500.0, 0.0],
                [0.0, 0.5]])
Q2 = np.array([[1.0, 0.0],
              [0.0, 8.0e-11]])

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



def plot_trace_Ex_Im(x_trace_list, u_trace_list, obj_list):
    all_grid_list, all_K_list = LoadData()
    x_trace_list_im = np.load('./ImplicitMPC_result/x_trace1_Im.npy', allow_pickle=True)
    u_trace_list_im = np.load('./ImplicitMPC_result/u_trace1_Im.npy', allow_pickle=True)
    obj_list_im = np.load('./ImplicitMPC_result/obj_trace1_Im.npy', allow_pickle=True)
    step = [i for i in range(1, len(x_trace_list))]
    
    init_x1 = x_trace_list[0][0]
    init_x2 = x_trace_list[0][1] 

    x1_list = [x[0] for x in x_trace_list]
    x2_list = [x[1] for x in x_trace_list]
    
    u1_list = [u[0] for u in u_trace_list]
    u2_list = [u[1] for u in u_trace_list]
    
    x1_list_im = [x[0] for x in x_trace_list_im]
    x2_list_im = [x[1] for x in x_trace_list_im]
    
    u1_list_im = [u[0] for u in u_trace_list_im]
    u2_list_im = [u[1] for u in u_trace_list_im]
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-80, 80, 0.1)
    x, y = np.meshgrid(x,y)
    plt.contour(x, y, 1060*x**2+44*x*y+0.52*y**2, [372], colors=['black']) #stability region

    plt.title('x_trajectory')
    plt.plot(x1_list, x2_list, c='r', label='Explicit MPC')
    plt.scatter(x1_list, x2_list, c='r', marker='o', s=15, label='Explicit MPC')
    plt.plot(x1_list_im, x2_list_im, c='g', label='Implicit MPC')
    plt.scatter(x1_list_im, x2_list_im, c='g', marker='*', s=15, label='Implicit MPC')
    
    plt.plot([0], [0], marker='*',markersize=5)
    plt.plot(init_x1, init_x2, marker='s',markersize=5)
    plt.xlim((-2,2))
    plt.ylim((-100,100))
    plt.xlabel('C$_A$-C$_A$$_s$')
    plt.ylabel('T-T$_s$')
    plt.grid()
    plt.legend()
    
    ax0 = plt.subplot(2,2,2)
    major_locator0 = plt.MultipleLocator(0.05)
    ax0.xaxis.set_major_locator(major_locator0)
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-80, 80, 0.1)
    x, y = np.meshgrid(x,y)
    plt.contour(x, y, 1060*x**2+44*x*y+0.52*y**2, [372], colors=['black']) #stability region
    plt.title('x_trajectory')
    plt.plot(x1_list, x2_list, c='r', label='Explicit MPC')
    plt.scatter(x1_list, x2_list, c='r', marker='o', s=15, label='Explicit MPC')
    plt.plot(x1_list_im, x2_list_im, c='g', label='Implicit MPC')
    plt.scatter(x1_list_im, x2_list_im, c='g', marker='*', s=15, label='Implicit MPC')
    plt.plot([0], [0], marker='*',markersize=5)
    plt.plot(init_x1, init_x2, marker='s',markersize=5)
    plt.xlim((-0.1, 0.1))
    plt.ylim((-4, 4))
    plt.xlabel('C$_A$-C$_A$$_s$')
    plt.ylabel('T-T$_s$')
    plt.grid()
    plt.legend()
    
    ax1 = plt.subplot(2,2,3)
    ax1.set_title('u_trajectory')
    u1_major_locator = plt.MultipleLocator(5)
    ax1.xaxis.set_major_locator(u1_major_locator)
    line1 = ax1.step(step, u1_list, c='red', linewidth=1, linestyle='-', where='post', label='Explicit MPC-u1')
    line2 = ax1.step(step, u1_list_im, c='g', linewidth=1, linestyle='-', where='post', label='Implicit MPC-u1')
    ax1.axis(ymin=-4.5,ymax=4.5)
    ax1.axhline(0, c='grey', linewidth=0.5)
    ax1.set_ylabel('u1')
    ax1.set_xlabel('step')

    ax2 = ax1.twinx()
    line3 = ax2.step(step, u2_list, c='red', linewidth=1, linestyle='--', where='post', label='Explicit MPC-u2')
    line4 = ax2.step(step, u2_list_im, c='g', linewidth=1, linestyle='--', where='post', label='Implicit MPC-u2')
    ax2.axis(ymin=-5.5e5,ymax=5.5e5)
    ax2.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax2.set_ylabel('u2')

    lns = line1+line2+line3+line4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    ax3 = plt.subplot(2,2,4)
    major_locator3 = plt.MultipleLocator(5)
    ax3.xaxis.set_major_locator(major_locator3)
    plt.title('obj_trajectory')
    plt.plot(step, obj_list, c='r', label='Explicit MPC')
    plt.plot(step, obj_list_im, c='g', label='Implicit MPC')
    plt.axhline(0, ls='--', c='black')
    plt.xlabel('step')
    plt.ylabel('obj_value')
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def eval_f(x_k1, x_k2, u_k, u_k1):
    global Q1, Q2
    x_k1_t = np.expand_dims(x_k1, axis=1)
    x_k2_t = np.expand_dims(x_k2, axis=1)
    u_k_t = np.expand_dims(u_k, axis=1)
    u_k1_t = np.expand_dims(u_k1, axis=1)
    x_k1 = np.expand_dims(x_k1, axis=2)
    x_k2 = np.expand_dims(x_k2, axis=2)
    u_k = np.expand_dims(u_k, axis=2)
    u_k1 = np.expand_dims(u_k1, axis=2)
    obj_list = x_k2_t@Q1@x_k2 + x_k1_t@Q1@x_k1 + u_k1_t@Q2@u_k1 + u_k_t@Q2@u_k
    return obj_list.reshape(-1,)

def GetExplicitMPC_sol(x_k, sols):
    x_k_all_list = []
    u_k_all_list = []
    u_k1_all_list = []
    obj_value_all_list = []

    for sol in sols:
        u_k_all = sol.evaluate(x_k.reshape(-1,1))
        if u_k_all is not None:
            u_k_all = u_k_all.reshape(-1,)
            u_k_all_list.append(u_k_all[:2])
            u_k1_all_list.append(u_k_all[2:])
            x_k_all_list.append(x_k)
    x_k_all_list = np.array(x_k_all_list)       
    u_k_all_list = np.array(u_k_all_list)
    u_k1_all_list = np.array(u_k1_all_list)

    x_k1_all_list, x_k2_all_list = FNN_next_state(x_k_all_list, u_k_all_list, u_k1_all_list)
    obj_value_all_list = eval_f(x_k1_all_list, x_k2_all_list, u_k_all_list, u_k1_all_list)

    
    min_loc = np.argmin(obj_value_all_list)
    best_u_k = u_k_all_list[min_loc].reshape(-1,)
    return best_u_k, obj_value_all_list[min_loc]

def FNN_next_state(x_k, u_k, u_k1):
    global input_scaler, output_scaler
    input_ = np.column_stack((x_k, u_k, u_k1))    
    input_total_ = input_scaler.transform(input_)
    input_total_ = input_total_.reshape(-1, 1, 6)
    
    x_k_ = torch.from_numpy(input_total_[:,:,:2]).to(torch.float32).to(device)
    u_k_ = torch.from_numpy(input_total_[:,:,2:4]).to(torch.float32).to(device)
    u_k1_ = torch.from_numpy(input_total_[:,:,4:]).to(torch.float32).to(device)

    x_k1_, x_k2_ = model(x_k_, u_k_, u_k1_)

    x_k1_ = x_k1_.cpu().data.numpy()[:,0]
    x_k2_ = x_k2_.cpu().data.numpy()[:,0]
    
    output_ = np.column_stack((x_k1_, x_k2_))
    output = output_scaler.inverse_transform(output_)
    x_k1 = output[:,:2]
    x_k2 = output[:,2:]
    return x_k1, x_k2

def FP_next_state(x_k, u_k):
    print(x_k, u_k)
    x_k1 = np.array([0.0, 0.0])
    
    dx1_dt = F/V*(-x_k[0]+(C_A0s-C_As)+u_k[0])-k_0*np.exp(-Ea/(R*(x_k[1]+T_s)))*(x_k[0]+C_As)**2
    dx2_dt = F/V*(-x_k[1]+(T_0-T_s))+(-drH)/(p_L*Cp)*k_0*np.exp(-Ea/(R*(x_k[1]+T_s)))*(x_k[0]+C_As)**2+(Q_s+u_k[1])/(p_L*Cp*V)

    dt = 0.01
    x_k1[0] = x_k[0] + dx1_dt*dt
    x_k1[1] = x_k[1] + dx2_dt*dt
    return x_k1

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

def plot_trace_scaled(x_trace_list):
    x1_list = [x[0] for x in x_trace_list]
    x2_list = [x[1] for x in x_trace_list]
    
    x_trace_list_im = np.load('./ImplicitMPC_result/x_trace1_scaled_Im.npy', allow_pickle=True)
    x1_list_im = [x[0] for x in x_trace_list_im]
    x2_list_im = [x[1] for x in x_trace_list_im]

    init_x1 = x_trace_list[0][0]
    init_x2 = x_trace_list[0][1]
    
    origin_x1, origin_x2 = input_scaler.transform([[0, 0, 0, 0, 0, 0]])[0][:2]
    
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    plt.title('x_trajectory with x1 space partition')
    plt.plot(x1_list, x2_list, c='r', linewidth=1, label='Explicit MPC')
    plt.scatter(x1_list, x2_list, c='r', marker='o', s=15, label='Explicit MPC')
    
    plt.plot(x1_list_im, x2_list_im, c='g', linewidth=1, label='Implicit MPC')
    plt.scatter(x1_list_im, x2_list_im, c='g', marker='*', s=15, label='Implicit MPC')
    lw = 0.5
    lc = 'grey'
    already_render_lines = set()
    for grid_pts in all_grid_list[0]:
        Lx1 = round(grid_pts[0][0], 6)
        Ux1 = round(grid_pts[1][0], 6)
        Lx2 = round(grid_pts[0][1], 6)
        Ux2 = round(grid_pts[2][1], 6)
        #LINE 1
        line_str = f'{Lx1}&{Ux1}&{Lx2}&{Lx2}'
        if line_str not in already_render_lines:
            x_list = np.array([Lx1, Ux1])
            y_list = np.array([Lx2, Lx2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
    
        #LINE 2
        line_str = f'{Ux1}&{Ux1}&{Lx2}&{Ux2}'
        if line_str not in already_render_lines:
            x_list = np.array([Ux1, Ux1])
            y_list = np.array([Lx2, Ux2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
        
        #LINE 3
        line_str = f'{Lx1}&{Ux1}&{Ux2}&{Ux2}'
        if line_str not in already_render_lines:
            x_list = np.array([Lx1, Ux1])
            y_list = np.array([Ux2, Ux2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
       
        #LINE 4
        line_str = f'{Lx1}&{Lx1}&{Lx2}&{Ux2}'
        if line_str not in already_render_lines:
            x_list = np.array([Lx1, Lx1])
            y_list = np.array([Lx2, Ux2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
    plt.plot([origin_x1], [origin_x2], marker='*',markersize=5)
    plt.plot(init_x1, init_x2, marker='s',markersize=5)

    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    
    plt.subplot(1,2,2)
    plt.title('x_trajectory with x2 space partition')
    plt.plot(x1_list, x2_list, c='r', linewidth=1, label='Explicit MPC')
    plt.scatter(x1_list, x2_list, c='r', marker='o', s=15, label='Explicit MPC')
    plt.plot(x1_list_im, x2_list_im, c='g', linewidth=1, label='Implicit MPC')
    plt.scatter(x1_list_im, x2_list_im, c='g', marker='*', s=15, label='Implicit MPC')
    already_render_lines = set()
    for grid_pts in all_grid_list[1]:
        Lx1 = round(grid_pts[0][0], 6)
        Ux1 = round(grid_pts[1][0], 6)
        Lx2 = round(grid_pts[0][1], 6)
        Ux2 = round(grid_pts[2][1], 6)
        #LINE 1
        line_str = f'{Lx1}&{Ux1}&{Lx2}&{Lx2}'
        if line_str not in already_render_lines:
            x_list = np.array([Lx1, Ux1])
            y_list = np.array([Lx2, Lx2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
    
        #LINE 2
        line_str = f'{Ux1}&{Ux1}&{Lx2}&{Ux2}'
        if line_str not in already_render_lines:
            x_list = np.array([Ux1, Ux1])
            y_list = np.array([Lx2, Ux2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
        
        #LINE 3
        line_str = f'{Lx1}&{Ux1}&{Ux2}&{Ux2}'
        if line_str not in already_render_lines:
            x_list = np.array([Lx1, Ux1])
            y_list = np.array([Ux2, Ux2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
       
        #LINE 4
        line_str = f'{Lx1}&{Lx1}&{Lx2}&{Ux2}'
        if line_str not in already_render_lines:
            x_list = np.array([Lx1, Lx1])
            y_list = np.array([Lx2, Ux2])
            plt.plot(x_list, y_list, c=lc, linewidth=lw)
            already_render_lines.add(line_str)
    plt.plot([origin_x1], [origin_x2], marker='*',markersize=5)
    plt.plot(init_x1, init_x2, marker='s',markersize=5)
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')    
    plt.show()
    
def GetXGridDistance(All_Explicit_Sols):
    grid_distance_dict = {}
    total_x_grid_cnt = len(All_Explicit_Sols)
    for i in range(total_x_grid_cnt):
        grid_1 = All_Explicit_Sols[i][0]
        Ux1_1 = round(grid_1[0], 6)
        Lx1_1 = round(-grid_1[1], 6)
        Ux2_1 = round(grid_1[2], 6)
        Lx2_1 = round(-grid_1[3], 6)

        cpt_x1_1 = round((Lx1_1 + Ux1_1)/2, 6)
        cpt_x2_1 = round((Lx2_1 + Ux2_1)/2, 6)
        distance_dict = {}
        for j in range(total_x_grid_cnt):
            grid_2 = All_Explicit_Sols[j][0]
            Ux1_2 = round(grid_2[0], 6)
            Lx1_2 = round(-grid_2[1], 6)
            Ux2_2 = round(grid_2[2], 6)
            Lx2_2 = round(-grid_2[3], 6)
            cpt_x1_2 = round((Lx1_2 + Ux1_2)/2, 6)
            cpt_x2_2 = round((Lx2_2 + Ux2_2)/2, 6)
            distance = ((cpt_x1_1-cpt_x1_2)**2+(cpt_x2_1-cpt_x2_2)**2)**0.5
            distance_dict[j] = round(distance, 8)
        distance_dict_sorted = sorted(distance_dict.items(), key=lambda x: x[1])
        grid_distance_dict[i] = distance_dict_sorted
    return grid_distance_dict
            
            
            
if __name__ == "__main__":
    print('Loading ML Nodel')
    model = torch.load('./Train_Model/FNN_CSTR_6D_FP.pkl').to(device)
    print('Loading Explicit ML-MPC Solutions...')
    All_Explicit_Sols = np.load('ExplicitML-MPC_Sols.npy', allow_pickle=True).tolist()
    input_scaler = np.load('./Train_Model/input_scaler_FP.npy', allow_pickle=True).item()
    output_scaler = np.load('./Train_Model/output_scaler_FP.npy', allow_pickle=True).item()
    all_grid_list, all_K_list = LoadData()

    #Get the weight matrix to implement NFS alorightm
    grid_distance_dict = GetXGridDistance(All_Explicit_Sols)

    #inition condition
    x1 = 1.0
    x2 = -50
    C_As = 1.95 
    T_s = 402.0

    
    x_k = np.array([x1, x2])
    x_trace_list = [np.array([x1, x2])]
    x_trace_list_scaled  = [input_scaler.transform([[x_k[0], x_k[1], 0, 0, 0, 0]])[0][:2]]
    
    u_trace_list = []
    obj_trace_list = []  
    time_st = time.time()

    #step 1, locate the initial region ID using sequential search algorithm
    cnt = 0
    for region_id, info in enumerate(All_Explicit_Sols):
        cnt += 1
        bds_x = np.array(info[0])
        if all(w_x@x_k <= bds_x):
            best_u_k, obj_value = GetExplicitMPC_sol(x_k, info[1])
            break 
    
    x_k1 = FP_next_state(x_k, best_u_k)
    x_trace_list_scaled.append(input_scaler.transform([[x_k[0], x_k[1], 0, 0, 0, 0]])[0][:2])
    u_trace_list.append(best_u_k)
    obj_trace_list.append(obj_value)
    x_trace_list.append(x_k1)
    x_k = x_k1

    #for the rest of the simulation, locate the initial region ID using NFS algorithm
    for i in range(1, 30):
        try:
            #print(f'i--{i+1}', end='\t')
            search_list = [x[0] for x in grid_distance_dict[region_id]]
            cnt = 0
            max_iter_cnt = len(search_list)
            for region_id_next in search_list:
                cnt += 1
                bds_x = np.array(All_Explicit_Sols[region_id_next][0])
                if all(w_x@x_k <= bds_x):
                    region_id = region_id_next
                    best_u_k, obj_value = GetExplicitMPC_sol(x_k, All_Explicit_Sols[region_id_next][1])
                    break
                if cnt == max_iter_cnt:
                    raise ValueError('suitable region not found!')
                
            x_k1 = FP_next_state(x_k, best_u_k)
            x_trace_list_scaled.append(input_scaler.transform([[x_k[0], x_k[1], 0, 0, 0, 0]])[0][:2])
            u_trace_list.append(best_u_k)
            obj_trace_list.append(obj_value)
            x_trace_list.append(x_k1)
            x_k = x_k1
        except:
            pass
    time_end = time.time()
    print(time_end-time_st)

    plot_trace_Ex_Im(x_trace_list, u_trace_list, obj_trace_list)
    plot_trace_scaled(x_trace_list_scaled)
