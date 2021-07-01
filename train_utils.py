from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

global batch_size, test_batch_size, epochs, lr, gamma, no_cuda, dry_run, seed, log_interval, save_model, device

device = torch.device("cuda:0")
use_cuda = True
batch_size = 64
test_batch_size = 1000
epochs = 50
lr = 0.001
gamma = 0.2
no_cuda = False
dry_run = False
seed = 1
log_interval = 1000
save_model = True
num_base_example = 15000


train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

#loss function for teacher
def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

#loss function for student
def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5, T=100):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def generate_const_T(idx = None):
    return 100


def generate_av_T(index, max_index = 44, max_T = 200, min_T = 1):
    if IS_CONST_T:
        return torch.Tensor([generate_const_T()])
    else:
        return ((max_T - min_T)/max_index * index).int()

def train(model, device, train_loader, optimizer, epoch, teacher,alpha=0.5):
    model.train()
    loss_list  = np.array([]) 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        teacher_output = teacher(data)
        T = generate_const_T()
        loss = loss_fn_kd(output, target, teacher_output,alpha = alpha, T = T)
       
        loss_list = np.append(loss_list, np.mean(loss.cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break
    return np.mean(loss_list)
        
def train_with_idx(model, device, train_loader, optimizer, epoch, teacher,alpha=0.5):
    model.train()
    loss_list  = np.array([]) 
    for batch_idx, (data, target, data_score) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        teacher_output = teacher(data)
        T = generate_av_T(data_score)
        loss = loss_fn_kd(output, target, teacher_output,alpha = alpha, T = int(T.float().mean().item()))
       
        loss_list = np.append(loss_list, np.mean(loss.cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break
    return np.mean(loss_list)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
        
def train_with_teacher(model, teacher, train_loader, test_loader, model_name = 'model', alpha=0.2, save_model = False, is_const_T= True):
    global IS_CONST_T
    IS_CONST_T = is_const_T
    acc_list = np.array([])
    loss_list = np.array([])
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    
    #model = models.resnet50().to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.Adadelta(model.parameters(), lr=lr) 
    
    #scheduler = StepLR(optimizer, step_size=50, gamma=gamma)

    for epoch in range(1, epochs + 1):  

        loss = train_with_idx(model, device, train_loader, optimizer, epoch, teacher, alpha = alpha)
        _, acc = test(model, device, test_loader)
        loss_list = np.append(loss_list, loss)
        acc_list = np.append(acc_list, acc)
        #scheduler.step()
    
    if save_model:
        if teacher is None:
            torch.save(model.state_dict(), f'{model_name}.pt')
        else:
            torch.save(model.state_dict(), f'{model_name}_a_{alpha}_t_{temperature}.pt')
    return loss_list, acc_list
