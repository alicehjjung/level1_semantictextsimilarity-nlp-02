import torch
import os
import numpy as np

def seed_everything(seed: int=1234):
    np.random.seed(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = '1'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    