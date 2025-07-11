from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import numpy as np

from sklearn.cluster import KMeans
from cluster_tools import kmeans_elbow

from utils import clone_model

import time

def calc_grad(model:nn.Module, data, target, loss_function):
    """
    Function to compute the grad
    args : data, target, loss_function
    return loss
    """
    outputs = model(data)
    loss = loss_function(outputs, target)
    loss.backward() #compute grad
    return loss

def solve_problem(model,
                  criterion,
                  optimizer_class,
                  dataloader:DataLoader,
                  n_epoch: int=100, 
                  time_lim=None, # in seconds
                  verbose=False):
    assert n_epoch or time_lim, "No limit to the number of iterations"

    if verbose:
        print("Building models...")
    optimizer = optimizer_class(model.parameters())

    n = len(dataloader.dataset)
    loss_hist = []
    timestamps = []
    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for i in tqdm(range(n_epoch)):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = calc_grad(model, batch, None, criterion)
            optimizer.step()
            epoch_loss += len(batch)/n*loss.item()
        
        loss_hist.append(epoch_loss)
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break
    
    return np.array(timestamps), np.array(loss_hist), model

def svrg(model:nn.Module,
         loss_function,
         dataloader:DataLoader,
         *model_args,
         n_epoch,
         time_lim=None,
         learning_rate=0.001, # for correspondance with torch.optim function 
         print_freq=None):
    """
    Function to updated weights with a SVRG backpropagation \\
    args : dataset, loss function, number of epochs, learning rate \\
    return : total_loss_epoch
    """
    assert n_epoch or time_lim, "No limit to the number of iterations"

    total_loss_epoch = []
    timestamps = []
    dataset = torch.from_numpy(dataloader.dataset)

    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for epoch in tqdm(range(n_epoch)):
        previous_net = clone_model(model, *model_args) # for calculating gradient each step
        
        #Compute full grad
        previous_net.zero_grad()
        total_loss_epoch.append(
            calc_grad(previous_net, dataset, None, loss_function).item())
        
        previous_net_grads = [p.grad.data for p in previous_net.parameters()]
        # print(total_loss_epoch[epoch])

        #Run over the dataset
        for batch in dataloader:

            #Compute prev stoc grad
            previous_net.zero_grad() #grad = 0
            prev_loss = calc_grad(previous_net, batch, None, loss_function)
            
            #Compute cur stoc grad
            model.zero_grad() #grad = 0
            cur_loss = calc_grad(model, batch, None, loss_function)
            
            #Backward
            for param1, param2, param3 in zip(model.parameters(), previous_net.parameters(), previous_net_grads): 
                param1.data.sub_((learning_rate) * (param1.grad.data - param2.grad.data + param3))
        
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break

    return np.array(timestamps), np.array(total_loss_epoch), model

def COVER(model:nn.Module,
         loss_function,
         data,
         sampler:Sampler,
         *model_args,
         n_epoch,
         time_lim=None,
         learning_rate=0.001, # for correspondance with torch.optim function
         ):
    """
    See COVER: a cluster-based variance reduced method for online learning (Yuan et al. 2019)
    """
    assert n_epoch or time_lim, "No limit to the number of iterations"

    total_loss_epoch = []
    timestamps = []
    dataset = torch.from_numpy(data)
    n = len(dataset)

    cluster_count = kmeans_elbow(dataset)
    kmeans_res = KMeans(n_clusters=cluster_count).fit(dataset)
    cluster_probs = np.zeros(cluster_count)
    for l in kmeans_res.labels_:
        cluster_probs[l] += 1
    cluster_probs /= np.sum(cluster_probs)

    relax = 0.0005 # np.min(cluster_probs)/2
    cluster_relax = relax/cluster_probs

    # total_l = calc_grad(model, dataset, None, loss_function) # for stability I think you need this
    g_cluster = [[torch.zeros_like(p) for p in model.parameters()] # p.grad.data.detach().clone()
                  for _ in range(cluster_count)]
    g_bar = [torch.zeros_like(p) for p in model.parameters()]

    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for epoch in tqdm(range(n_epoch)):
        running_loss = 0
        # shuffled_ids = torch.randperm(n)
        for data_id in sampler:
            batch = dataset[data_id]
            model.zero_grad()
            loss = calc_grad(model, batch, "", loss_function)
            running_loss += loss.item() / n
            # print(running_loss, *model.parameters())
            curr_cluster = 0
            curr_cluster = kmeans_res.labels_[data_id]
            
            for param, g_c, g_b in zip(model.parameters(),
                                       g_cluster[curr_cluster],
                                       g_bar):
                grad = param.grad.data
                param.data.sub_(learning_rate*(grad - g_c + g_b))
                g_c.add_(cluster_relax[curr_cluster]*(g_c - grad))
                g_b.add_(relax*(g_c - grad))
        
        total_loss_epoch.append(running_loss)
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break

    return np.array(timestamps), np.array(total_loss_epoch), model

def clusterSVRG(model:nn.Module,
         loss_function,
         data,
         sampler:Sampler,
         *model_args,
         n_epoch,
         time_lim=None,
         learning_rate=0.001, # for correspondance with torch.optim function
         ):
    """
    See https://arxiv.org/abs/1602.02151
    """
    assert n_epoch or time_lim, "No limit to the number of iterations"

    total_loss_epoch = []
    timestamps = []
    dataset = torch.from_numpy(data)
    n = len(dataset)

    cluster_count = kmeans_elbow(dataset)
    kmeans_res = KMeans(n_clusters=cluster_count).fit(dataset)
    cluster_probs = np.zeros(cluster_count)
    for l in kmeans_res.labels_:
        cluster_probs[l] += 1
    cluster_probs /= np.sum(cluster_probs)


    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for epoch in tqdm(range(n_epoch)):
        #Compute full grad
        previous_net = clone_model(model, *model_args) # for calculating gradient each step
        previous_net.zero_grad()
        total_loss_epoch.append(
            calc_grad(previous_net, dataset, None, loss_function).item())
        previous_net_grads = [p.grad.data for p in previous_net.parameters()]

        z_cluster = [[torch.zeros_like(p) for p in model.parameters()]
                    for _ in range(cluster_count)]
        z_total = [torch.zeros_like(p) for p in model.parameters()]
        for data_id in sampler:
            batch = dataset[data_id]

            previous_net.zero_grad()
            prev_loss = calc_grad(previous_net, batch, "", loss_function)

            model.zero_grad()
            curr_loss = calc_grad(model, batch, "", loss_function)
            
            curr_cluster = kmeans_res.labels_[data_id]
            for param, g_c, g_c_total, p_prev, g_total, in \
                    zip(model.parameters(),
                        z_cluster[curr_cluster],
                        z_total,
                        previous_net.parameters(), 
                        previous_net_grads):
                grad = param.grad.data
                prev_grad = p_prev.grad.data
                param.data.sub_(learning_rate*(g_total + g_c_total + grad - (g_c + prev_grad)))
                g_c_total.add_(cluster_probs[curr_cluster]*(grad - prev_grad - g_c))
                g_c = grad - prev_grad
        
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break

    return np.array(timestamps), np.array(total_loss_epoch), model