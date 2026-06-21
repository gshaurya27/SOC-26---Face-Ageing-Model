import torch 
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis =0 )
x= np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy :', outputs)

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x,dim=0)
print('softmax tensor:',outputs)

def cross_entropy(actual,predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss

#y must be one hot encoded - i.e. should be in 0s and 1s 
y = np.array([1.,0,0])

#y_pred has probabilities 
y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(y,y_pred_good)
l2 = cross_entropy(y, y_pred_bad)

print(f'loss1: {l1:.4f}')
print(f'loss2: {l2:.4f}')

loss=nn.CrossEntropyLoss()
# y not be one hot coded 
Y = torch.tensor([2,0,1])
Y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]])
Y_pred_bad = torch.tensor([[2.1,1.0,0.1],[1.0,1.0,2.1],[3.1,0.1,2.1]])

L1 = loss(Y_pred_good,Y)
L2 = loss(Y_pred_bad,Y)

print(L1.item())
print(L2.item())

_, predictions1 = torch.max(Y_pred_good,1)
_, predictions2 = torch.max(Y_pred_bad,1)
print(predictions1)
print(predictions2)
