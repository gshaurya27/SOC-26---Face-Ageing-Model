import torch
import numpy as np

x = torch.ones(2, 2, dtype=torch.float32)
print(x)
print(x[:,0]) # prints the first column
print(x[1, 1].item()) #only prints the value in 1,1 place
print(x.size)


y=torch.rand(2 ,2)
print(y)
y.add_(x)

z= torch.tensor([2.5, 0.1])
print(z)

n = x*y
m = x + y
print(m)

x1 = x.view(-1,1) #automatically checks the number of elements and select the number of appropriate rows
x2 = x.numpy()  # creates a numpy array that points to the same memory location
x3 = torch.from_numpy(x2) #creates a tensor that has same memory location as numpy 

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device) # to move this to GPU
    z = x+y # this will be performed in the GPU
    z = z.to("cpu") # to move this back to CP
    

