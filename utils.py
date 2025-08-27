import copy
from torch.nn import Module

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0]*size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        else: return x
    
    def union(self, x, y):
        r"""also returns True / False for "changes made"""

        x_rep = self.find(x)
        y_rep = self.find(y)

        if x_rep == y_rep: 
            return False
        
        if self.rank[x_rep] < self.rank[y_rep]:
            self.parent[x_rep] = y_rep
        else:
            self.parent[y_rep] = x_rep
            if self.rank[x_rep] == self.rank[y_rep]:
                self.rank[x_rep] += 1
        return True

def clone_model(model:Module, *args):
    # Create a new instance of the same class
    model_copy = type(model)(*args)
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    return model_copy

def decimal_part(x):
    return x - int(x)

def classlookup(cls):
    c = list(cls.__bases__)
    for base in c:
        c.extend(classlookup(base))
    return c


def dimension_reduction(X, dim=2, alg="pca"):
    if X.shape[1] <= dim: return X 
    if alg == "pca":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) 
        pca = PCA(n_components=dim)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca
    elif alg == "tsne":
        tsne = TSNE(dim)
        tsne_result = tsne.fit_transform(X)
        return tsne_result
    else: 
        raise NotImplementedError("Only 'pca' or 'tsne' are allowed")