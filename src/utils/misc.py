import random
import numpy as np
import torch

def get_device_available():
    ''' Detect available training device'''
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def set_seed(seed):
    ''' Set random seed '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device_available()
    if device == torch.device("mps"):
        torch.mps.manual_seed(seed)
    elif device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)