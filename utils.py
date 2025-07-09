import copy
from torch.nn import Module

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