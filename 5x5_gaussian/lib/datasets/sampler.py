import torch
import numpy as np

class UniformSampler(object):
    def __init__(self, dim):
        self.dim = dim
        self.name = "uniform"
    
    def sampling(self, n):
        return torch.rand(n, self.dim)

class GaussianSampler(object):
    def __init__(self, dim):
        self.dim = dim
        self.name = "gaussian"
    
    def sampling(self, n):
        return torch.randn(n, self.dim)


def loadSampler(sampler_config, dataset=None):
    sampler_name = sampler_config['name']
    if sampler_name == "uniform":
        return UniformSampler(sampler_config['dim'])
    elif sampler_name == "gaussian":
        return GaussianSampler(sampler_config['dim'])
    else:
        raise ValueError("no such sampler called " + sampler_name)
