from math import sqrt
from numpy import dot, argmax, zeros

def fast_norm(x):
    """
    Returns norm-2 of a 1-D numpy array.
    """
    return sqrt(dot(x, x.T))

def compet(x):
    """
    Returns a 1-D numpy array with the same size as x,
    where the element having the index of the maximum value in x has value 1, others have value 0.
    """
    idx = argmax(x)
    res = zeros((len(x)))
    res[idx] = 1
    return(res)

def euclidean_distance(a, b):
    """
    Returns Euclidean distance between 2 vectors.
    """
    x = a - b
    return(fast_norm(x))

def default_bias_function(biases, win_idx):
    """
    Default function that reduces bias value of winner neuron and increases bias value of other neurons
    """
    biases = biases * 0.9
    biases[win_idx] = biases[win_idx] / 0.9 - 0.2
    return biases

def default_non_bias_function(biases, win_idx):
    return biases

def default_learning_rate_decay_function(learning_rate, iteration, decay_rate):
    return learning_rate / (1 + decay_rate * iteration)

def default_radius_decay_function(sigma, iteration, decay_rate):
    return sigma / (1 + decay_rate * iteration)
