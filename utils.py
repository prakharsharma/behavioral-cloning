"""
Utility functions
"""
import numpy as np


# helper function to normalize data
def normalize(x):
    x = (x - np.mean(x))/np.std(x)
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    x -= 0.5
    return x