# PyTorch tutorial codes for course EL-7143 Advanced Machine Learning, NYU, Spring 2019
# main.py: trainig neural networks for MNIST classification
import time, datetime
from options import parser
from models import ConvNet, FCNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms

def get_data_loader(args):
    ''' define training and testing data loader'''
    # load trainig data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./Data/mnist', train=True, download=True, 
                               transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    # load testing data loader
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./Data/mnist', train=False, 
                               transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)
    
    return train_loader, test_loader

def get_model(args):
    ''' define model '''
    model = None
    if args.fc:
            model = FCNet()
    else:
            model = ConvNet()
    if args.cuda:
            model.cuda()
            
    print('\n---Model Information---')
    print('Net:',model)
    print('Use GPU:', args.cuda)
    
    return model
	
def get_optimizer(args, model):
    ''' define optimizer '''
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print('\n---Training Details---')
    print('batch size:',args.batch_size)
    print('seed number', args.seed)

    print('\n---Optimization Information---')
    print('optimizer: SGD')
    print('lr:', args.lr)
    print('momentum:', args.momentum)
    
    return optimizer

def train(model, optimizer, train_loader, epoch):
    ''' define training function '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    ''' define testing function '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
        
if __name__ == '__main__':
    start_time = datetime.datetime.now().replace(microsecond=0)
    print('\n---Started training at---', (start_time))
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train_loader, test_loader = get_data_loader(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, epoch)
        test(model, test_loader)
        current_time = datetime.datetime.now().replace(microsecond=0)
        print('Time Interval:', current_time - start_time, '\n')
