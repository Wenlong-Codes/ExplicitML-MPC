# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:37:27 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import sleep
import gc
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class SimDataset(Dataset):
    def __init__(self, trj_data):
        trj_data = np.expand_dims(trj_data, axis=1)
        self.input_k = trj_data[:,:,:6]
        self.output_k1_k2 = trj_data[:,:,6:]

    def __len__(self):
        return int(len(self.input_k))

    def __getitem__(self, index):
        x_k = self.input_k[index][:,:2]
        u_k = self.input_k[index][:,2:4]
        u_k1 = self.input_k[index][:,4:]
        
        x_k1 = self.output_k1_k2[index][:,:2]
        x_k2 = self.output_k1_k2[index][:,2:]
        
        return x_k, u_k, u_k1, x_k1, x_k2



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


def load_sim_data(file_name):
    sim_data_dict = np.load(f'../Open-loop simulation/{file_name}.npy', allow_pickle=True).tolist()
    C_A_input = np.array(sim_data_dict[0])
    T_input = np.array(sim_data_dict[1])
    C_A_output = np.array(sim_data_dict[2])
    T_output = np.array(sim_data_dict[3])
    C_A0_input = np.array(sim_data_dict[4])
    Q_input = np.array(sim_data_dict[5])
    del sim_data_dict
    gc.collect()
    
    # x1_k, x2_k, u1_k, u2_k, u1_k1, u2_k1
    input_k = np.stack((C_A_input, T_input, C_A0_input[:,0], Q_input[:,0], C_A0_input[:,1], Q_input[:,1]), axis=1)
    input_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(input_k)
    input_k_scaled = input_scaler.transform(input_k)
    
    # x1_k1, x2_k1, x1_k2, x2_k2
    output_k1_k2 = np.stack((C_A_output[:,0], T_output[:,0], C_A_output[:,1], T_output[:,1]), axis=1)
    output_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(output_k1_k2)
    output_k1_k2_scaled = output_scaler.transform(output_k1_k2)
    trj_data_scaled = np.column_stack((input_k_scaled, output_k1_k2_scaled))
    np.save('input_scaler.npy', input_scaler)
    np.save('output_scaler.npy', output_scaler)

    
    return np.round(trj_data_scaled, 8)


def model_evaluation(model, dl):
    model.eval()
    output_list = []
    for i, (x_k, u_k, u_k1, x_k1, x_k2) in enumerate(dl):
        x_k = x_k.to(torch.float32).to(device)
        u_k = u_k.to(torch.float32).to(device)
        u_k1 = u_k1.to(torch.float32).to(device)
        x_k1 = x_k1.to(torch.float32).to(device)
        x_k2 = x_k2.to(torch.float32).to(device)
        x_k1_pred, x_k2_pred = model(x_k, u_k, u_k1)
    
        label = torch.cat((x_k1, x_k2), axis=1)
        pred = torch.cat((x_k1_pred, x_k2_pred), axis=1)
        loss = loss_fn(label, pred)
        output_list.append(loss.item())
    loss_ave = np.mean(np.array(output_list))
    return loss_ave


if __name__ == "__main__": 
    # x=[x1, x2], u=[u1, u2], x_hat=[x1_hat, x2_hat]
    
    LOAD = 1
    if LOAD:
        trj_data_train = np.load('trj_data_train.npy', allow_pickle=True)
        trj_data_valid = np.load('trj_data_valid.npy', allow_pickle=True)
        trj_data_test = np.load('trj_data_test.npy', allow_pickle=True)
    else:
        trj_data = load_sim_data('FP_sim_6D') 
        trj_data_train, trj_data_valid = train_test_split(trj_data, test_size=0.1, random_state=123)
        trj_data_train, trj_data_test = train_test_split(trj_data_train, test_size=0.1, random_state=456)
        np.save('trj_data_train.npy', trj_data_train)
        np.save('trj_data_valid.npy', trj_data_valid)
        np.save('trj_data_test.npy', trj_data_test)
        print('all save!')
        
    train_ds = SimDataset(trj_data_train)
    valid_ds = SimDataset(trj_data_valid)
    test_ds = SimDataset(trj_data_test)
    train_dl = DataLoader(train_ds, batch_size=2**12, shuffle=True, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=2**12, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=2**12, shuffle=False, drop_last=False)
    
    model = ML_Model().to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        output_list = []
        for i, (x_k, u_k, u_k1, x_k1, x_k2) in enumerate(tqdm(train_dl)):
            x_k = x_k.to(torch.float32).to(device)
            u_k = u_k.to(torch.float32).to(device)
            u_k1 = u_k1.to(torch.float32).to(device)
            x_k1 = x_k1.to(torch.float32).to(device)
            x_k2 = x_k2.to(torch.float32).to(device)
            
            x_k1_pred, x_k2_pred = model(x_k, u_k, u_k1)
            label = torch.cat((x_k1, x_k2), axis=1)
            pred = torch.cat((x_k1_pred, x_k2_pred), axis=1)
            loss = loss_fn(label, pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_list.append(loss.item())
        loss_ave = np.mean(np.array(output_list))
        tqdm.write(f'epoch: {epoch+1}\tloss_ave: {loss_ave}')
        if (epoch+1) % 10 == 0:
            valid_MSE = model_evaluation(model, valid_dl)
            print(f'valid_MSE: {valid_MSE}')
            
    train_MSE = model_evaluation(model, train_dl)
    test_MSE = model_evaluation(model, test_dl)
    valid_MSE = model_evaluation(model, valid_dl)
    print(f'train_MSE: {train_MSE}')
    print(f'valid_MSE: {valid_MSE}')
    print(f'test_MSE: {test_MSE}')
    torch.save(model, 'FNN_CSTR_6D.pkl')
