# PyTorch tutorial codes for course EL-7143 Advanced Machine Learning, NYU, Spring 2019
from __future__ import print_function
from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#---------- Generate Target Polynomial Function ----------#
# A polynomial function: y = w_1* x^1 + w_2* x^2 + ... + w_n* x^n + b
# Learnable parameters: w_1, w_2, ..., w_n, b
# n = POLY_DEGREE
POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) 
b_target = torch.randn(1)

#---------- Define Auxiliary Functions ----------#
def make_features(x):
    '''Builds features i.e. a matrix with columns [x, x^2, x^3, x^4].'''
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

def f(x):
    '''Approximated function.'''
    return x.mm(W_target) + b_target[0]

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result

def get_batch_poly(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)

def get_batch_nn(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    x = torch.randn(batch_size, 1)
    x_poly = make_features(x.squeeze())
    y = f(x_poly)
    return Variable(x), Variable(y)

#---------- Define Model ----------#
# Linear Model 
model_poly = torch.nn.Linear(W_target.size(0), 1)
optimizer_poly = optim.SGD(model_poly.parameters(), lr = 0.1, momentum=0.0)

# Non-Linear Model 
_model_nn  = [torch.nn.Linear(1, 256)]
_model_nn += [torch.nn.ReLU(True)]
_model_nn += [torch.nn.Linear(256, 1)]
model_nn = torch.nn.Sequential(*_model_nn)
optimizer_nn = optim.SGD(model_nn.parameters(), lr = 0.1, momentum=0.0)

#---------- Train ----------#
# Linear Model
for batch_idx_poly in count(1):

    # Generate training data
    batch_x, batch_y = get_batch_poly()              

    # Forward pass
    optimizer_poly.zero_grad()
    output = model_poly(batch_x)                     
    loss = F.smooth_l1_loss(output, batch_y)  
    loss_data = loss.item()

    # Backward pass
    loss.backward() 
    optimizer_poly.step() 

    # Stop iteration
    if loss_data < 1e-3:  
        break

# Non-Linear Model
for batch_idx_nn in count(1):

    # Generate training data
    batch_x, batch_y = get_batch_nn()              

    # Forward pass
    optimizer_nn.zero_grad()
    output = model_nn(batch_x)                   
    loss = F.smooth_l1_loss(output, batch_y)  
    loss_data = loss.item()

    # Backward pass
    loss.backward() 
    optimizer_nn.step() 

    # Stop iteration
    if loss_data < 1e-3:  
        break

#---------- Print ----------#
# Linear Model
batch_x, batch_y = get_batch_poly(1000)
output = model_poly(batch_x)                     
loss_data = F.smooth_l1_loss(output, batch_y).item()
print('-----Linear approximation-----')
print('Loss: {:.6f} after {} batches'.format(loss_data, batch_idx_poly))
print('==> Learned function:\t' + poly_desc(model_poly.weight.data.view(-1), model_poly.bias.data))
print('==>  Actual function:\t' + poly_desc(W_target.view(-1), b_target))

# Non-Linear Model
batch_x, batch_y = get_batch_nn(1000)
output = model_nn(batch_x)                     
loss_data = F.smooth_l1_loss(output, batch_y).item()
print('-----Non-Linear approximation-----')
print('Loss: {:.6f} after {} batches'.format(loss_data, batch_idx_nn))
