from kneed import KneeLocator
from sklearn.cluster import KMeans
import numpy as np

from torch.utils.data import Sampler
from collections.abc import Sized, Iterator

from sklearn.metrics.pairwise import cosine_similarity

def cosine_distance(x, y):
    # Cosine similarity
    return 1 - cosine_similarity([x], [y])[0][0]

# KMeans++ initialization using cosine distance from ChatGPT
def kmeans_plus_plus_cosine(X, k):
    # X is the dataset with shape (n_samples, n_features)
    # k is the number of clusters
    
    # Step 1: Choose the first centroid randomly from the data points
    centroids = [X[np.random.randint(0, len(X))]]
    
    for _ in range(1, k):
        # Step 2: Compute distances from each point to the closest centroid
        distances = []
        for point in X:
            # Compute the minimum distance to any centroid
            min_dist = min(cosine_distance(point, centroid) for centroid in centroids)
            distances.append(min_dist)
        
        # Step 3: Choose the next centroid with probability proportional to the squared distance
        distances = np.array(distances)
        prob_distances = distances**2
        prob_distances /= prob_distances.sum()  # Normalize to get probabilities
        
        # Select the next centroid randomly based on the computed probabilities
        next_centroid_idx = np.random.choice(len(X), p=prob_distances)
        centroids.append(X[next_centroid_idx])
    
    return np.array(centroids)

# KMeans with Cosine Distance - Full Algorithm (slightly modified from GPT)
def kmeans_cosine(X, k, max_iters=100, tol=1e-4):
    # Initialize centroids using KMeans++ with cosine distance
    centroids = kmeans_plus_plus_cosine(X, k)
    inertia = 0
    for _ in range(max_iters):
        inertia = 0
        
        # Step 1: Assign each point to the nearest centroid based on cosine distance
        labels = []
        for point in X:
            distances = [cosine_distance(point, centroid) for centroid in centroids]
            label = np.argmin(distances)  # Find the closest centroid
            labels.append(label)
        labels = np.array(labels)
        
        # Step 2: Update the centroids
        new_centroids = []
        for i in range(k):
            # Compute the mean of the points assigned to this centroid
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroid = np.mean(points_in_cluster, axis=0)
            else:
                new_centroid = centroids[i]  # Keep the old centroid if no points are assigned
            new_centroids.append(new_centroid)
        
        new_centroids = np.array(new_centroids)
        
        # Step 3: Check for convergence (if centroids don't change significantly)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        for i, point in enumerate(X):
            inertia += cosine_distance(point, new_centroids[labels[i]])

        # Update centroids for the next iteration
        centroids = new_centroids
    
    return centroids, labels, inertia

def kmeans_elbow(dataset, distance="euclidian", cap=25):
    lim = min(int(np.sqrt(len(dataset))), cap)
    K = range(1, lim)
    inertias = []
    if distance == "euclidian":
        for k in K:
            kmeans = KMeans(n_clusters=k).fit(dataset)
            inertias.append(kmeans.inertia_)
    elif distance == "cosine":
        for k in K:
            _, _, inertia = kmeans_cosine(dataset, k)
            inertias.append(inertia)
    else:
        raise NotImplementedError("Only 'euclidian' or 'cosine'")
    kl = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    return kl.elbow

def kmeans_pp_elbow(dataset):
    ideal_k = kmeans_elbow(dataset)
    kmeans = KMeans(n_clusters=ideal_k).fit(dataset)
    return kmeans.labels_

def kmeans_cosine_elbow(dataset):
    ideal_k = kmeans_elbow(dataset, distance="cosine")
    _, labels, _ = kmeans_cosine(dataset, ideal_k)
    return labels

class ClusterSampler(Sampler):
    def __init__(self, 
                 data_source:Sized, 
                 batch_size, 
                 clustering_method=kmeans_pp_elbow):
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