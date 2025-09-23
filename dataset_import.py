
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
import torchvision
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

from sklearn.datasets import load_svmlight_file
import warnings

def make_2d_classification(N=4000, well_split=True):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    a_pos1 = torch.randn(N//4, 2) + torch.tensor([3.0, 2.0])
    a_pos2 = torch.randn(N//4, 2) + torch.tensor([-3.0, 3.0])
    if well_split:
        a_neg = torch.randn(N//2, 2) + torch.tensor([-2.0, -5.0])
    else:
        a_neg = torch.randn(N//2, 2) + torch.tensor([-2.0, -2.0])
    data = torch.cat([a_pos1, a_pos2, a_neg], dim=0)
    data = (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0))
    labels = torch.cat([torch.ones(N//2), torch.zeros(N//2)])
    return TensorDataset(data, labels)

def MNIST_01(train=True):
    full_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=train,
                                              download=True, 
                                              )
    # Get data and targets
    data = full_dataset.data.flatten(start_dim=1)
    targets = full_dataset.targets

    # Create mask
    mask = (targets == 0) | (targets == 1)
    
    scaler = StandardScaler()
    filtered_data = torch.from_numpy(scaler.fit_transform(data[mask])).float()

    # filtered_data = (filtered_data - filtered_data.mean(axis=0)) / np.sqrt(filtered_data.var(axis=0))

    label_map = {0: -1, 1: 1}
    filtered_targets = torch.tensor([label_map[t.item()] for t in targets[mask]])

    return TensorDataset(filtered_data, filtered_targets)



def blob_dataset():
    DATASET_SIZE = 1000
    DATASET_DIM = 20
    DATASET_CLUSTER_COUNT = 5
    clustered_data, _ = make_blobs(n_samples=DATASET_SIZE, 
                                   n_features=DATASET_DIM,
                                   centers=DATASET_CLUSTER_COUNT,
                                   random_state=42
                                   )

    data = torch.from_numpy(clustered_data.astype(np.float32))
    labels = torch.zeros(DATASET_SIZE)
    return TensorDataset(data, labels)

def read_csv_tensordataset(filename, sep=";"):
    df = pd.read_csv(filename, sep=sep)
    X = torch.from_numpy(df.iloc[:, :-1].values).float()
    y = torch.from_numpy(df.iloc[:, -1].values).float()
    return TensorDataset(X, y)



def load_svm_classif_tensor(filename, process=None):
    X, y = load_svmlight_file(filename)
    X = torch.tensor(X.toarray(), dtype=torch.float32)

    scaler = StandardScaler()
    scaled_X = torch.from_numpy(scaler.fit_transform(X)).float()

    y = torch.tensor(y, dtype=torch.int32)
    if process is not None: # converts the other possible label types to 0/1
        if process == "1 2":
            y = 2*y - 3
        elif process == "0 1":
            y = 2*y - 1
        else:
            warnings.warn("process should be one of '1 2' or '0 1'", UserWarning)
    return TensorDataset(scaled_X, y)

def load_dataset(dataset_name, **kwargs):
    if dataset_name == "simple_2d":
        return make_2d_classification(**kwargs)
    elif dataset_name == "MNIST_01":
        return MNIST_01(**kwargs)
    elif dataset_name == "phishing":
        return load_svm_classif_tensor("./data/phishing_dataset.txt", "0 1")
    elif dataset_name == "white_wine":
        return read_csv_tensordataset("./data/winequality-white.csv")