from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np

import torch
from torch.utils.data import Sampler, TensorDataset
from collections.abc import Sized, Iterator

from sklearn.cluster import kmeans_plusplus
import matplotlib.pyplot as plt

import faiss
import heapq
from utils import UnionFind

def display_clusters(data, labels, k, centroids=None, alpha=0.5):
    for i in range(k):
        plt.scatter(data[labels==i,0], data[labels==i,1], alpha=alpha)
    if centroids is not None:
        plt.scatter(centroids[:,0], centroids[:,1], color="black")

def alt_kmeans(X, k, max_iter=100, tol=1e-4):
    centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=42)

    for it in range(max_iter):
        # d[k, l] = dist between X[k,:] and C[l,:]
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
        if it == 0:
            labels = np.argmin(distances, axis=1)
        else:
            labels = np.argmin(distances*n_elem, axis=1)
        n_elem = np.bincount(labels, minlength=k)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        # Update centroids for the next iteration
        centroids = new_centroids
    return labels

def kmeans_elbow(dataset, cap=25):
    lim = min(int(np.sqrt(len(dataset))), cap)
    K = range(1, lim)
    inertias = []
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(dataset)
        inertias.append(kmeans.inertia_)
    
    kl = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    return kl.elbow

def kmeans_pp_elbow(dataset, distance="euclidian"):
    if distance == "euclidian":
        data = dataset
    elif distance == "cosine":
        data = preprocessing.normalize(dataset)
    else:
        raise NotImplementedError("Only 'euclidian' or 'cosine'") 
    ideal_k = kmeans_elbow(data)
    kmeans = KMeans(n_clusters=ideal_k).fit(data)
    
    return kmeans.labels_

kmeans_cos_elbow = lambda data: kmeans_pp_elbow(data, "cosine")

def logistic_label_01_process(dataset:TensorDataset):
    data, labels = dataset.tensors
    res = data * (1 - 2*labels).unsqueeze(1).float()
    return res

def logistic_label_pm1_process(dataset:TensorDataset):
    data, labels = dataset.tensors
    res = data * labels.unsqueeze(1).float()
    return res

def get_clusters(dataset: TensorDataset, 
                 label_processing=None, 
                 clustering_method=kmeans_pp_elbow):
    if label_processing is None:
            data_source = dataset.tensors[0]
    else:
        data_source = label_processing(dataset)

    labels = clustering_method(data_source)
    return labels

class ClusterSampler(Sampler):
    def __init__(self, 
                 num_samples,
                 batch_size, 
                 labels):
        
        self.cluster_count = max(labels)+1
        self.cluster_sizes = [0] * self.cluster_count
        for l in labels:
            self.cluster_sizes[l] += 1
        
        self.clusters = [torch.zeros(self.cluster_sizes[i]).int() for i in range(self.cluster_count)]
        counts = [0] * self.cluster_count
        for i, l in enumerate(labels):
            self.clusters[l][counts[l]] = i
            counts[l] += 1

        self.num_samples = num_samples
        self.batch_size = batch_size

        self.sample_count = [self.cluster_sizes[i]*batch_size//self.num_samples
                             for i in range(self.cluster_count)]

        self.added_count = self.batch_size - sum(self.sample_count)

        # we miss num_samples % batch_size elements in the process so we add them back
        self.last_batch_size = self.num_samples%self.batch_size
        self.last_sample_count = [len(self.clusters[i])*self.last_batch_size//self.num_samples
                                  for i in range(self.cluster_count)]
        self.last_added_count = self.last_batch_size - sum(self.last_sample_count)

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples//self.batch_size):
            for i_cluster in range(self.cluster_count):
                selected_ids = torch.randint(self.cluster_sizes[i_cluster], (self.sample_count[i_cluster],))
                yield from self.clusters[i_cluster][selected_ids]
            
            yield from torch.randint(self.num_samples, (self.added_count,))
        
        for i_cluster in range(self.cluster_count):
            selected_ids = torch.randint(self.cluster_sizes[i_cluster], (self.last_sample_count[i_cluster],))
            yield from self.clusters[i_cluster][selected_ids]
        
        yield from torch.randint(self.num_samples, (self.last_added_count,))


def approx_nearest_clustering(vectors, edge_count = 16, knn=10, wanted_cluster_count=8):
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, edge_count)
    # index.hnsw.efConstruction = 40 # Controls construction time/accuracy tradeoff
    index.add(vectors)

    # building edges
    distances, neighbors = index.search(vectors, knn)
    size = len(vectors)
    edges = []
    for i in range(size):
        for d, n in zip(distances[i], neighbors[i]):
            if d == 0: continue
            heapq.heappush(edges, (d, (i, n)))
    
    # building the clusters
    uf = UnionFind(size)
    curr_cluster_count = size
    while curr_cluster_count > wanted_cluster_count and len(edges) > 0:
        _, (x, y) = heapq.heappop(edges)
        if uf.union(x, y):
            curr_cluster_count -= 1
    
    # retrieving the clusters
    cluster_counter = 0
    labels = np.ones(size, dtype=np.int32) * (-1)
    cluster_ids = dict()
    for i in range(size):
        elem = uf.find(i)
        if elem not in cluster_ids:
            cluster_ids.update({elem: cluster_counter})
            cluster_counter += 1
        labels[i] = cluster_ids[elem]
    
    return labels