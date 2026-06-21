# 1) Design Model (input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
#       - forward pass : compute prediction
#       - backward pass : gradients
#       - update weights

import torch
import torch.nn as nn

#f=w*x (linear regression, weights into input)

#f=2*x
#Data for training we have as X and Y
# for prediction with torch we need different shape for X and Y that is each row becomes a sample and columns becomes features
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features


# model is just internally creating the forwad funtion we made earlier
# we dont want weights and bias to be made earlier and set to 0, it creates two tensors for w and b of length input, output respeactively 
# internally it makes the operationas Y = Xw + b that is the matrix multiplication
# model.parameters() contains those two w and b parameters

#model = nn.Linear(input_size, output_size)

class LinearRegression(nn.module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define leayers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

#model can have only tensor as input and not float value
print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

#Training 
learning_rate = 0.0001
n_iters = 1000000

# a loss function created that can be called, which calculates mean square error ( that is basicaaly what we were doing)
loss = nn.MSELoss()

# This creates a Stochastic Gradient Descent optimizer that will optimize w, that is equivalent to the same thing we were g=doing before instead of optimizer.step()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #predicting forward
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = Backwardpass
    l.backward() 

    #update weights
    optimizer.step()

    #grad to zero 
    optimizer.zero_grad()

    if epoch % 150000 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')