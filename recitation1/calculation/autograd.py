# PyTorch tutorial codes for course EL-7143 Advanced Machine Learning, NYU, Spring 2019
import torch
import torch.nn.functional as F

# 32-bit floating point
# floating-point arithmetic: https://en.wikipedia.org/wiki/Floating-point_arithmetic
dtype = torch.float32 
# put tensor on cpu(or you can try GPU)
device = torch.device("cpu")

# our data
x = [[1], [2]]
W1 = [[1,1], [2, 1]]
W2 = [[1, 2]]

x = torch.tensor(x, dtype=dtype, device=device, requires_grad = True)
W1 = torch.tensor(W1, dtype=dtype, device=device, requires_grad = True)
W2 = torch.tensor(W2, dtype=dtype, device=device, requires_grad = True)

# our function
def f1(x):
    return x**2

def f2(x):
    return x*2

# forward
y1 = W1.mm(x)
y2 = f1(y1)
y3 = W2.mm(y2)
y4 = f2(y3)

# backward
y_hat = y4.clone().detach().requires_grad_(False)
y_hat.add_(1)
loss = 1/2* F.mse_loss(y4, y_hat)
print("--The gradient of y is [%.2f]"%(2*loss.item()))
loss.backward()

# check gradients
print("--The gradient calculated by PyTorch is:")
print(x.grad.data)
print("--The gradient calculated by us is:")
our_grad = -(W1.t()* (2*y1.t())).mm(W2.t())* 2
print(our_grad.data)
