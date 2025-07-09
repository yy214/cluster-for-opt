# slightly modified from https://github.com/kilianFatras/variance_reduced_neural_networks

import numpy as np
import copy
import torch
from torch.nn import Module
from torch.autograd import Variable, grad
from tqdm import tqdm
from utils import clone_model

def calc_grad(model:Module, data, target, loss_function):
    """
    Function to compute the grad
    args : data, target, loss_function
    return loss
    """
    outputs = model(data)
    loss = loss_function(outputs, target)
    loss.backward() #compute grad
    return loss


def svrg(model:Module, dataset, loss_function, n_epoch, learning_rate, print_freq=None, *model_args):
    """
    Function to updated weights with a SVRG backpropagation \\
    args : dataset, loss function, number of epochs, learning rate \\
    return : total_loss_epoch
    """
    total_loss_epoch = [0] * n_epoch

    n_samples = len(dataset)

    for epoch in tqdm(range(n_epoch)):
        running_loss = 0.0
        previous_net = clone_model(model, *model_args) # for calculating gradient each step
        
        #Compute full grad
        previous_net.zero_grad()
        total_loss_epoch[epoch] \
            = calc_grad(previous_net, dataset, loss_function)
        
        previous_net_grads = [p.grad.data for p in previous_net.parameters()]
        print(total_loss_epoch[epoch])

        #Run over the dataset
        for i_data, data in enumerate(dataset): # more like dataloader
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels) #wrap data and target into variable
            
            #Compute prev stoc grad
            previous_net.zero_grad() #grad = 0
            prev_loss = calc_grad(previous_net, inputs, labels, loss_function)
            
            #Compute cur stoc grad
            model.zero_grad() #grad = 0
            cur_loss = calc_grad(model, inputs, labels, loss_function)
            
            #Backward
            for param1, param2, param3 in zip(model.parameters(), previous_net.parameters(), previous_net_grads): 
                param1.data -= (learning_rate) * (param1.grad.data - param2.grad.data + (1./n_samples) * param3)

            # print statistics
            running_loss += cur_loss.data[0]
            if print_freq and i_data % print_freq == print_freq-1:    # print every 2500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i_data + 1, running_loss / print_freq))
                running_loss = 0.0
            
    return total_loss_epoch

