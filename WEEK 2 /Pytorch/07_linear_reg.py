# 1) Design Model (input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
#       - forward pass : compute prediction
#       - backward pass : gradients
#       - update weights

import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets  #to generate regression dataset 
import matplotlib.pyplot as plt

# 0) Prepare Data 
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100, #rows of arrays 
    n_features = 1, # columns 
    noise=20, # make more realistic
    random_state=1  # makes the dataset same every tym we run the code
    )

    # this creates random arrays which follow y = wx + b + noise  

X=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
#this y is currently just a single row vector so convert to column vector of different shape(vecotr in vector)
y=y.view(y.shape[0],1)

n_samples, n_features = X.shape

# 1)model

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2)loss and optimizer

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# 3)training loop 

num_epoch = 100

for epoch in range(num_epoch):
    #forward pass
    y_pred = model(X)
    #loss
    loss = criterion(y_pred, y)
    #backward pass
    loss.backward()
    #updated weights 
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch%15 ==0:
        print(f'epoch{epoch+1}, loss = {loss.item():.3f}')

#plot 
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro") #(x-axis, y-axis, red dots)
plt.plot(X_numpy, predicted, "b")   #(x-axis, y-axis, blue line)
plt.show()
