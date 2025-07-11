from kneed import KneeLocator
from sklearn.cluster import KMeans
import numpy as np

from torch.utils.data import Sampler
from collections.abc import Sized, Iterator

def kmeans_elbow(dataset, cap=25):
    lim = min(int(np.sqrt(len(dataset))), cap)
    K = range(1, lim)
    inertias = []

    for k in K:
        kmeans = KMeans(n_clusters=k).fit(dataset)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    return kl.elbow

class ClusterSampler(Sampler):
    def __init__(self, data_source:Sized, batch_size):
        self.cluster_count = kmeans_elbow(data_source)
        kmeans_res = KMeans(n_clusters=self.cluster_count).fit(data_source)
        self.clusters = [[] for _ in range(self.cluster_count)]
        for i, l in enumerate(kmeans_res.labels_):
            self.clusters[l].append(i)

        self.num_samples = len(data_source)
        self.batch_size = batch_size

        self.sample_count = [len(self.clusters[i])*batch_size//self.num_samples
                             for i in range(self.cluster_count)]

        self.added_count = self.batch_size - sum(self.sample_count)
        self.added_probs =  np.array([len(self.clusters[i])*batch_size%self.num_samples
                             for i in range(self.cluster_count)], dtype=np.float64)
        self.added_probs /= np.sum(self.added_probs)

        # we miss num_samples % batch_size elements in the process so we add them back
        self.last_batch_size = self.num_samples%self.batch_size
        self.last_sample_count = [len(self.clusters[i])*self.last_batch_size//self.num_samples
                                  for i in range(self.cluster_count)]
        self.last_added_count = self.last_batch_size - sum(self.last_sample_count)
        self.last_added_probs =  np.array([len(self.clusters[i])*self.last_batch_size%self.num_samples
                             for i in range(self.cluster_count)], dtype=np.float64)
        self.last_added_probs /= np.sum(self.last_added_probs)

    def __iter__(self) -> Iterator[int]:
        
        for _ in range(self.num_samples//self.batch_size):
            added = np.random.choice(self.cluster_count, self.added_count, p=self.added_probs)
            sample_count = self.sample_count[:]
            for i_added in added:
                sample_count[i_added] += 1
            
            for i_cluster in range(self.cluster_count):
                selected = np.random.choice(self.clusters[i_cluster], sample_count[i_cluster])
                yield from selected
                
        added = np.random.choice(self.cluster_count, self.last_added_count, p=self.last_added_probs)
        sample_count = self.last_sample_count[:]
        for i_added in added:
            sample_count[i_added] += 1
        
        for i_cluster in range(self.cluster_count):
            selected = np.random.choice(self.clusters[i_cluster], sample_count[i_cluster])
            yield from selected