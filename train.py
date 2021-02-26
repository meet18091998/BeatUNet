#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import dill
import time

from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import logging
from model import BeatUNet

TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")

BATCH_SIZE = 2
EPS = 1e-12

dataset_path = './'
dataset_X = dill.load(open(dataset_path+'padded_max_logmels', 'rb'))
dataset_Y = dill.load(open(dataset_path+'padded_frameset', 'rb'))

dataset_X = dataset_X.reshape((dataset_X.shape[0], 1, dataset_X.shape[1], dataset_X.shape[2]))

X_train, y_train = dataset_X[:5000], dataset_Y[:5000].long()
X_test, y_test = dataset_X[5000:], dataset_Y[5000:].long()

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)
val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)
dataloaders = {'train':train_loader, 'validation':val_loader}

NUM_EPOCHS = 5000

LEARNING_RATE = 1e-3

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CUDA = True


cuda = 0
device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

model = BeatUNet()

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

is_better = True
prev_acc = 0

liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath='./figures/training.png')])

def acc(test_Y, pred_y_l) :
    c = 0
    for i in range(len(pred_y_l)) :
        for j in range(len(pred_y_l[0])) :
            if test_Y[i,j] == pred_y_l[i,j] :
                c += 1
    return (c/(i*j))

for epoch in range(NUM_EPOCHS):
    logs = {}
    t_start = time.time()
    
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
            
        else:
            model.eval()
        model.to(device)
        
        print("Started Phase")

        running_loss = 0.0
        
        predicted_phase = torch.zeros(len(dataloaders[phase].dataset), 5, 4821)
        target_phase = torch.zeros(len(dataloaders[phase].dataset), 4821)
        
        if phase == 'validation':
            
            with torch.no_grad():
                
                for (i,batch) in enumerate(dataloaders[phase]):
                    
                    input_tensor = batch[0]
                    input_tensor = input_tensor.to(device)
                    bs = input_tensor.shape[0]
                    target_tensor = batch[1].long().to(device)

                    softmaxed_tensor = model(input_tensor)

                    loss = criterion(softmaxed_tensor, target_tensor)

                    predicted_phase[i*bs:(i+1)*bs] = softmaxed_tensor.cpu()
                    target_phase[i*bs:(i+1)*bs] = target_tensor.cpu()


                    input_tensor = input_tensor.cpu()
                    target_tensor = target_tensor.cpu()

                    running_loss += loss.detach() * input_tensor.size(0)
                    
        else:
            
            for (i,batch) in enumerate(dataloaders[phase]):
                input_tensor = batch[0]
                input_tensor = input_tensor.to(device)
                bs = input_tensor.shape[0]
                target_tensor = batch[1].long().to(device)

                softmaxed_tensor = model(input_tensor)
         
                loss = criterion(softmaxed_tensor, target_tensor)
            
                predicted_phase[i*bs:(i+1)*bs] = softmaxed_tensor.cpu()
                target_phase[i*bs:(i+1)*bs] = target_tensor.cpu()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                input_tensor = input_tensor.cpu()
                target_tensor = target_tensor.cpu()

                running_loss += loss.detach() * input_tensor.size(0)
    

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_predicted = torch.argmax(predicted_phase, dim=1)
        epoch_accuracy = acc(target_phase, epoch_predicted)
        
        
        model.to('cpu')

        prefix = ''
        if phase == 'validation':
            prefix = 'val_'

        logs[prefix + 'log loss'] = epoch_loss.item()
        logs[prefix + 'accuracy'] = epoch_accuracy
        
        print('Phase time - ',time.time() - t_start)

    delta = time.time() - t_start
    is_better = logs['val_accuracy'] > prev_acc
    if is_better:
        prev_acc = logs['val_accuracy']
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss': logs['val_log loss'], 'accuracy': logs['val_accuracy']}, "./models/unet_pretrain_"+TIME_STAMP+"_"+str(logs['val_accuracy'])+".pth")


    liveloss.update(logs)
    liveloss.send()
