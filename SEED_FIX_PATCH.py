"""
SEED_FIX_PATCH.py
-----------------
Drop-in utility to add deterministic seeding to any FedUCA notebook cell.
Call set_all_seeds(seed) at the top of every loop iteration that initialises
a model or runs local training.

Usage:
    from SEED_FIX_PATCH import set_all_seeds
    for seed in SEEDS:
        set_all_seeds(seed)
        head = BottleneckMLP(ratio=4).to(DEVICE)
        ...
"""
import torch, numpy as np, random, os

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism on GPU (may slow training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)

