# 1) Design Model (input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
#       - forward pass : compute prediction
#       - backward pass : gradients
#       - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets #to generate regression dataset 
from sklearn.preprocessing import StandardScaler #to scale features 
from sklearn.model_selection import train_test_split # to have separation of training and testing data
import matplotlib.pyplot as plt

# 0) Data

bc = datasets.load_breast_cancer()      # already available data is used 
X, y = bc.data, bc.target       # this data is then splited into input(data) and output (target)

n_samples, n_features = X.shape  

X_train , X_test, y_train , y_test = train_test_split(X, y , 
                                                      test_size=0.2         #this means that 20% for testing and 80 for training 
                                                      , random_state=1234)

#scale (done for making things in between 0 and 1 like if age varies btw 18 to 90 this will be reduced to 0 to 1)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)
# fit finds two values that are u=mean and s=stnd deviation
# transformation converts x to x'=((x-u)/s)
#if fit again used in the test set information will get leaked adn also we want to use the same u ans s

X_train = torch.from_numpy(X_train.astype(np.float32))      #conversion from numpy to torch data 
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# f=wx+b, sigmoid at end 

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)


# 2) loss and optimizer 
learning_rate = 0.0001
criterion = nn.BCELoss()        #Binary Cross Entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop
num_epoch = 200
for epoch in range(num_epoch):
    #forward
    y_pred = model(X_train)
    #loss
    loss = criterion(y_pred, y_train)
    #backward
    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) %15 == 0:
        print(f'epoch {epoch+1}:, loss= {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()         #converts to 0s and 1s
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0]) 
    print(f'accuracy = {acc:.4f}')