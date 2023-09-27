import random
import numpy as np
import torch
import os

def random_seed(seed=42, rank=0):
	torch.manual_seed(seed + rank)
	np.random.seed(seed + rank)
	random.seed(seed + rank)
	os.environ['PYTHONHASHSEED'] = str(seed + rank)
	np.random.seed(seed + rank)
	torch.manual_seed(seed + rank)
	torch.cuda.manual_seed(seed + rank)
	torch.cuda.manual_seed_all(seed + rank)
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True