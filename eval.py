
import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import dill
from model import BeatUNet

from sklearn.metrics import classification_report,  confusion_matrix

MODEL_PATH = "<replace_model_path>"

model = BeatUNet()
model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
model = model.to('cuda')

dataset_X = dill.load(open('padded_max_logmels', 'rb'))
dataset_Y = dill.load(open('padded_frameset', 'rb'))

dataset_X = dataset_X.reshape((dataset_X.shape[0], 1, dataset_X.shape[1], dataset_X.shape[2]))

train_X, train_Y = dataset_X[:5000], dataset_Y[:5000].long()
test_X, test_Y = dataset_X[5000:], dataset_Y[5000:].long()

del dataset_X, dataset_Y

t_y = [0]*(len(test_X))
# t_y = []
for i in range(len(test_X)) :
    batch = test_X[i:i+1].to('cuda')
    t_y[i] = model(batch).to('cpu').detach().numpy()[0]
    batch.to('cpu')
    if i%100 == 99 :
        print(i+1, end=' ')
    if i%1000 == 999:
        print()
        
pred_y = torch.tensor(t_y)
pred_y_l = torch.tensor([[torch.argmax(pred_y[i,:,j]) for j in range(len(pred_y[i,0,:]))] for i in range(len(pred_y)) ])

print("==============Test Classification Report===================")
print(classification_report(test_Y.flatten(), pred_y_l.flatten()))

print("==============Test Confusion Matrix===================")
print(confusion_matrix(test_Y.flatten(), pred_y_l.flatten()))
