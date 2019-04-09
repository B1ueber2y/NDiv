# ----------------------------------------
# Normalized Diversification
# NDiv loss implemented in Pytorch 
# ----------------------------------------
import numpy as np
import torch
import torch.nn.functional as F

def compute_pairwise_distance(x):
    ''' computation of pairwise distance matrix
    ---- Input
    - x: input tensor		torch.Tensor [(bs), sample_num, dim_x]
    ---- Return
    - matrix: output matrix	torch.Tensor [(bs), sample_num, sample_num]
    '''
    if len(x.shape) == 2:
        matrix = torch.norm(x[:,None,:] - x[None,:,:], p = 2, dim = 2)
    elif len(x.shape) == 3:
        matrix = torch.norm(x[:,:,None,:] - x[:,None,:,:], p = 2, dim = 3)
    else:
        raise NotImplementedError
    return matrix

def compute_norm_pairwise_distance(x):
    ''' computation of normalized pairwise distance matrix
    ---- Input
    - x: input tensor		torch.Tensor [(bs), sample_num, dim_x]
    ---- Return
    - matrix: output matrix	torch.Tensor [(bs), sample_num, sample_num]
    '''
    x_pair_dist = compute_pairwise_distance(x)
    normalizer = torch.sum(x_pair_dist, dim = -1)
    x_norm_pair_dist = x_pair_dist / (normalizer[...,None] + 1e-12).detach()
    return x_norm_pair_dist

def NDiv_loss(z, y, alpha=0.8):
    ''' NDiv loss function.
    ---- Input
    - z: latent samples after embedding h_Z:		torch.Tensor [(bs), sample_num, dim_z].
    - y: corresponding outputs after embedding h_Y:	torch.Tensor [(bs), sample_num, dim_y].
    - alpha: hyperparameter alpha in NDiv loss.
    ---- Return
    - loss: normalized diversity loss.			torch.Tensor [(bs)]
    '''
    S = z.shape[-2] # sample number
    y_norm_pair_dist = compute_norm_pairwise_distance(y)
    z_norm_pair_dist = compute_norm_pairwise_distance(z)
    ndiv_loss_matrix = F.relu(z_norm_pair_dist * alpha - y_norm_pair_dist)
    ndiv_loss = ndiv_loss_matrix.sum(-1).sum(-1) / (S * (S - 1))
    return ndiv_loss



