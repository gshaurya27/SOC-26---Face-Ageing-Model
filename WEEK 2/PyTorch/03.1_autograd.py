import torch

x = torch.rand(3 , requires_grad=True)
print(x)

y=x+2
print(y)

z=y*y*2
print(z)
#z=z.mean()
print(z)

v = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32) # these are the weights if z is scalar weight is automatically taken to be 1

z1 = z.backward(v) #dz/dx is computed in this line and the z is changed to dz/dx matrix
print(x.grad)
print(z1)

# if we that the gradient may not be tracked we can type the following
#x.requires_grad_(False)
#y =  x.detach(), here y doesnt require gradient
with torch.no_grad():
    n = x*2
    print(n)
