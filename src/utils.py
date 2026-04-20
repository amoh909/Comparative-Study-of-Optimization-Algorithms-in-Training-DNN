import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """Sets all seeds to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # These two lines ensure deterministic behavior on GPU (at a small speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed set to: {seed}")

def get_device(): ## To run on GPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")