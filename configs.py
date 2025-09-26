from dataclasses import dataclass
from typing import Optional, Callable, Type
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.optim.optimizer import Optimizer
from torch.optim import SGD
from simple_model import least_squares_crit, log_criterion
from cluster_tools import logistic_label_pm1_process, no_process

@dataclass
class Config:

    dataset_name: str

    n_epoch: Optional[int]=100
    time_lim: Optional[float]=None # 25
    batch_size: int=64
    repeat_times: int=10

    l2: float=1e-4
    bias: bool=True

    lr: float=2
    lr_lambda: Callable[[float], float]=lambda epoch: 5 / (epoch+1)
    optimizer_class: Type[Optimizer]=SGD
    criterion: Callable[[Tensor, Tensor], Tensor]=log_criterion
    processing: Callable[[TensorDataset, Tensor], Tensor]=logistic_label_pm1_process


# learning rate for phishing: lr=0.2

configs = {
    "simple_2d": Config("simple_2d"),
    "MNIST_01": Config("MNIST_01"),
    "phishing": Config("phishing", lr=0.2),
    "white_wine": Config("white_wine", 
                         lr=8e-5, 
                         bias=False,
                         criterion=least_squares_crit,
                         processing=no_process),
    "simple_reg": Config("simple_reg", lr=0.1, l2=0)
}