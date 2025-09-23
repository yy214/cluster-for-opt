import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, dim: int, bias: bool):
        super().__init__()
        self.layer = nn.Linear(dim, 1, bias=bias)
        self.layer.weight.data.fill_(0)
        if bias:
            self.layer.bias.data.fill_(0)
    
    def forward(self, ai):
        return self.layer(ai)

def least_squares_crit(ax: torch.Tensor, b: torch.Tensor):
    if ax.shape[0] != 1:
        raise ValueError(f"Expected shape (1, n) for ax, got {ax.shape}")
    return (1/2*((ax.flatten() - b)**2)).mean()

def log_criterion(pred: torch.Tensor, labels: torch.Tensor):
    l = labels.view(-1, 1)
    return (nn.functional.softplus(-l*pred)).mean()