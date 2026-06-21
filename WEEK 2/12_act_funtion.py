import torch 
import torch.nn as nn 
import numpy as np

#option 1
class NN(nn.Module):
    def __init__(self , input_size , hidden_size):
        super(NN , self).__init__()
        self.linear1 = nn.linear(input_size , hidden_size)
        self.relu = nn.ReLU()
        # nn.Sigmoid , nn.softmax , nn.TanH , nn.LeakyRelu
        self.linear2 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.sigmoid

    def forward (self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
#option 2
class NeuNet(nn.Module):
    def __init__(self , input_size , hidden_size):
        super(NeuNet,self).__init__()
        self.linear1 = nn.linear(input_size,hidden_size)
        self.linear2 = nn.linear(hidden_size,1)

    def forward(self ,x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear(out))
        return out