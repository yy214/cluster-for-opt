from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np

from torch.utils.data import Sampler, TensorDataset
from collections.abc import Sized, Iterator

from sklearn.metrics.pairwise import cosine_similarity


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

# TODO: optimize this (possible)
class ClusterSampler(Sampler):
    def __init__(self, 
                 dataset: TensorDataset, 
                 batch_size, 
                 label_processing=None,
                 clustering_method=kmeans_pp_elbow):
        
        if label_processing is None:
            data_source = dataset.tensors[0]
        else:
            data_source = label_processing(dataset)

        labels = clustering_method(data_source)
        self.cluster_count = max(labels)+1
        self.clusters = [[] for _ in range(self.cluster_count)]
        for i, l in enumerate(labels):
            self.clusters[l].append(i)

        self.num_samples = len(data_source)
        self.batch_size = batch_size

        self.sample_count = [len(self.clusters[i])*batch_size//self.num_samples
                             for i in range(self.cluster_count)]

        self.added_count = self.batch_size - sum(self.sample_count)
        self.added_probs =  np.array([len(self.clusters[i])*batch_size%self.num_samples
                             for i in range(self.cluster_count)], dtype=np.float64)
        remain = np.sum(self.added_probs)
        if remain != 0:
            self.added_probs /= remain
        else:
            self.added_probs = np.ones(self.cluster_count) / self.cluster_count

        # we miss num_samples % batch_size elements in the process so we add them back
        self.last_batch_size = self.num_samples%self.batch_size
        self.last_sample_count = [len(self.clusters[i])*self.last_batch_size//self.num_samples
                                  for i in range(self.cluster_count)]
        self.last_added_count = self.last_batch_size - sum(self.last_sample_count)
        self.last_added_probs =  np.array([len(self.clusters[i])*self.last_batch_size%self.num_samples
                             for i in range(self.cluster_count)], dtype=np.float64)
        
        last_remain = np.sum(self.last_added_probs)
        if last_remain != 0:
            self.last_added_probs /= last_remain
        else:
            self.last_added_probs = np.ones(self.cluster_count) / self.cluster_count

    def __iter__(self) -> Iterator[int]:
        
        for _ in range(self.num_samples//self.batch_size):
            added = np.random.choice(self.cluster_count, self.added_count, p=self.added_probs, replace=False)
            sample_count = self.sample_count[:]
            for i_added in added:
                sample_count[i_added] += 1
            
            for i_cluster in range(self.cluster_count):
                selected = np.random.choice(self.clusters[i_cluster], sample_count[i_cluster])
                yield from selected
                
        added = np.random.choice(self.cluster_count, self.last_added_count, p=self.last_added_probs, replace=False)
        sample_count = self.last_sample_count[:]
        for i_added in added:
            sample_count[i_added] += 1
        
        for i_cluster in range(self.cluster_count):
            selected = np.random.choice(self.clusters[i_cluster], sample_count[i_cluster])
            yield from selected