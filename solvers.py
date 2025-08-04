from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

import numpy as np

from sklearn.cluster import KMeans
from cluster_tools import kmeans_pp_elbow, get_clusters

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

def get_regularized_loss(model, data, labels, criterion, l2):
    output = model(data)
    loss = criterion(output, labels)
    reg_loss = loss + l2/2*sum((p**2).sum() for p in model.parameters())
    return reg_loss

def solve_problem(model,
                  criterion,
                  optimizer_class,
                  dataloader:DataLoader,
                  n_epoch: int=100, 
                  time_lim=None, # in seconds
                  verbose=False, 
                  lr=0.001,
                  lr_lambda=None,
                  l2=0):
    assert n_epoch or time_lim, "No limit to the number of iterations"

    if verbose:
        print("Building models...")
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=l2)

    if lr_lambda is not None:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    n = len(dataloader.dataset)
    loss_hist = []
    timestamps = []
    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for i in tqdm(range(n_epoch)):
        # epoch_loss = 0
        for (batch, labels) in dataloader:
            optimizer.zero_grad()
            loss = calc_grad(model, batch, labels, criterion)
            optimizer.step()
            # epoch_loss += len(batch)/n*loss.item()
        
        if lr_lambda is not None:
            scheduler.step()

        loss_hist.append(get_regularized_loss(model, 
                                              dataloader.dataset.tensors[0], 
                                              dataloader.dataset.tensors[1],
                                              criterion,
                                              l2).item())
        elapsed_t = time.perf_counter() - begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break
    
    return np.array(timestamps), np.array(loss_hist), model



def weighted_solver(model,
                    criterion,
                    optimizer,
                    datasource:TensorDataset,
                    n_iter: int=100000,
                    cluster_method=kmeans_pp_elbow, 
                    time_lim=None, # in seconds
                    verbose=False,
                    lr_lambda=None,
                    ):
    assert n_iter or time_lim, "No limit to the number of iterations"

    # if verbose:
        # print("Building models...")
    # optimizer = optimizer_class(model.parameters())

    if lr_lambda is not None:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    n = len(datasource)
    dataset, labels = datasource.tensors

    labels = cluster_method(dataset)
    cluster_count = max(labels)+1
    clusters = [[] for _ in range(cluster_count)]
    cluster_sizes = np.zeros(cluster_count)
    for i, l in enumerate(labels):
        clusters[l].append(i)
        cluster_sizes[l] += 1
    cluster_weights = torch.from_numpy(cluster_sizes) / n
    
    loss_hist = []
    timestamps = []
    begin_t = time.perf_counter()
    if n_iter is None:
        n_iter = 100000
    for i in tqdm(range(n_iter)):
        # choosing by batch
        batch_ids = [0]*cluster_count
        for i, c in enumerate(clusters):
            batch_ids[i] = np.random.choice(c)
        
        batch = dataset[batch_ids]
        label_batch = labels[batch_ids]
        outputs = model(batch)
        weighted = cluster_weights*outputs
        loss = criterion(weighted, label_batch)
        optimizer.zero_grad()
        loss.backward() #compute grad
        optimizer.step()

        l = len(batch)*loss.item()
        
        if lr_lambda is not None:
            scheduler.step()

        loss_hist.append(l)
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
         print_freq=None,
         lr_lambda=None,
         l2=0):
    """
    TODO: fix l2 regularization \\
    slightly modified from https://github.com/kilianFatras/variance_reduced_neural_networks \\
    Function to updated weights with a SVRG backpropagation \\
    args : dataset, loss function, number of epochs, learning rate \\
    return : total_loss_epoch
    """
    assert n_epoch or time_lim, "No limit to the number of iterations"

    total_loss_epoch = []
    timestamps = []
    dataset, labels = dataloader.dataset.tensors

    lr = learning_rate

    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for epoch in tqdm(range(n_epoch)):
        if lr_lambda is not None:
            lr = learning_rate*lr_lambda(epoch)

        previous_net = clone_model(model, *model_args) # for calculating gradient each step
        
        #Compute full grad
        previous_net.zero_grad()
        total_loss_epoch.append(
            calc_grad(previous_net, dataset, labels, loss_function).item())
        
        previous_net_grads = [p.grad.data for p in previous_net.parameters()]
        # print(total_loss_epoch[epoch])

        #Run over the dataset
        for (batch, label_batch) in dataloader:

            #Compute prev stoc grad
            previous_net.zero_grad() #grad = 0
            prev_loss = calc_grad(previous_net, batch, label_batch, loss_function)
            
            #Compute cur stoc grad
            model.zero_grad() #grad = 0
            cur_loss = calc_grad(model, batch, label_batch, loss_function)
            
            #Backward
            for param1, param2, param3 in zip(model.parameters(), previous_net.parameters(), previous_net_grads): 
                param1.data.sub_(lr * (param1.grad.data - param2.grad.data + param3 + l2*param1.data))
        
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break

    return np.array(timestamps), np.array(total_loss_epoch), model



# TODO: fix convergence
def COVER(model:nn.Module,
         loss_function,
         data:TensorDataset,
         sampler:Sampler,
         *model_args,
         n_epoch,
         cluster_labels,
         time_lim=None,
         learning_rate=0.001, # for correspondance with torch.optim function
         lr_lambda=None,
         l2=0,
         ):
    """
    TODO: fix l2 regularization \\
    See COVER: a cluster-based variance reduced method for online learning (Yuan et al. 2019)
    """
    assert n_epoch or time_lim, "No limit to the number of iterations"

    total_loss_epoch = []
    timestamps = []
    dataset, targets = data.tensors
    n = len(dataset)

    cluster_count = max(cluster_labels)+1
    cluster_probs = np.zeros(cluster_count)
    for l in cluster_labels:
        cluster_probs[l] += 1
    cluster_probs /= np.sum(cluster_probs)

    relax = 0.0005 # np.min(cluster_probs)/2
    cluster_relax = relax/cluster_probs

    # total_l = calc_grad(model, dataset, None, loss_function) # for stability I think you need this
    g_cluster = [[torch.zeros_like(p) for p in model.parameters()] # p.grad.data.detach().clone()
                  for _ in range(cluster_count)]
    g_bar = [torch.zeros_like(p) for p in model.parameters()]

    lr = learning_rate

    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for epoch in tqdm(range(n_epoch)):
        if lr_lambda is not None:
            lr = learning_rate*lr_lambda(epoch)

        running_loss = 0
        
        for data_id in sampler:
            batch = dataset[data_id]
            target_batch = targets[data_id]
            model.zero_grad()
            loss = calc_grad(model, batch, target_batch, loss_function)
            running_loss += loss.item() / n
            
            curr_cluster = 0
            curr_cluster = cluster_labels[data_id]
            
            for param, g_c, g_b in zip(model.parameters(),
                                       g_cluster[curr_cluster],
                                       g_bar):
                grad = param.grad.data
                param.data.sub_(lr*(grad - g_c + g_b + l2*param.data))
                g_c.add_(cluster_relax[curr_cluster]*(g_c - grad))
                g_b.add_(relax*(g_c - grad))
        
        total_loss_epoch.append(running_loss)
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break

    return np.array(timestamps), np.array(total_loss_epoch), model

#TODO: fix
def clusterSVRG(model:nn.Module,
         loss_function,
         data:TensorDataset,
         sampler:Sampler,
         *model_args,
         n_epoch,
         cluster_labels,
         time_lim=None,
         learning_rate=0.001, # for correspondance with torch.optim function
         lr_lambda=None
         ):
    """
    TODO: fix l2 regularization \\
    See https://arxiv.org/abs/1602.02151
    """
    assert n_epoch or time_lim, "No limit to the number of iterations"

    total_loss_epoch = []
    timestamps = []
    dataset, targets = data.tensors
    n = len(dataset)

    cluster_count = max(cluster_labels)+1
    cluster_probs = np.zeros(cluster_count)
    for l in cluster_labels:
        cluster_probs[l] += 1
    cluster_probs /= np.sum(cluster_probs)

    lr = learning_rate

    begin_t = time.perf_counter()
    if n_epoch is None:
        n_epoch = 100000
    for epoch in tqdm(range(n_epoch)):
        if lr_lambda is not None:
            lr = learning_rate*lr_lambda(epoch)

        #Compute full grad
        previous_net = clone_model(model, *model_args) # for calculating gradient each step
        previous_net.zero_grad()
        total_loss_epoch.append(
            calc_grad(previous_net, dataset, targets, loss_function).item())
        previous_net_grads = [p.grad.data for p in previous_net.parameters()]

        z_cluster = [[torch.zeros_like(p) for p in model.parameters()]
                    for _ in range(cluster_count)]
        z_total = [torch.zeros_like(p) for p in model.parameters()]
        for data_id in sampler:
            batch = dataset[data_id]
            target_batch = targets[data_id]

            previous_net.zero_grad()
            prev_loss = calc_grad(previous_net, batch, target_batch, loss_function)

            model.zero_grad()
            curr_loss = calc_grad(model, batch, target_batch, loss_function)
            
            curr_cluster = cluster_labels[data_id]
            for param, g_c, g_c_total, p_prev, g_total, in \
                    zip(model.parameters(),
                        z_cluster[curr_cluster],
                        z_total,
                        previous_net.parameters(), 
                        previous_net_grads):
                grad = param.grad.data
                prev_grad = p_prev.grad.data
                param.data.sub_(lr*(g_total + g_c_total + grad - (g_c + prev_grad)))
                g_c_total.add_(cluster_probs[curr_cluster]*(grad - prev_grad - g_c))
                g_c = grad - prev_grad
        
        elapsed_t = time.perf_counter()-begin_t
        timestamps.append(elapsed_t)
        if time_lim and elapsed_t > time_lim:
            break

    return np.array(timestamps), np.array(total_loss_epoch), model