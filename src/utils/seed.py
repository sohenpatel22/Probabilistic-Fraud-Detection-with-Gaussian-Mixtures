import numpy as np
import random


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """

    np.random.seed(seed)
    random.seed(seed)

    print(f"Seed set to: {seed}")