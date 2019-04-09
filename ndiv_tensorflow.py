# ----------------------------------------
# Normalized Diversification
# NDiv loss implemented in Tensorflow
# ----------------------------------------
import tensorflow as tf
import numpy as np

def compute_pairwise_distance(x):
    ''' computation of pairwise distance matrix
    ---- Input
    - x: input tensor		tf.Tensor [(bs), sample_num, dim_x]
    ---- Return
    - matrix: output matrix	tf.Tensor [(bs), sample_num, sample_num]
    '''
    x_shape = x.get_shape().as_list()
    if len(x_shape) == 2:
        matrix = tf.norm(x[:,None,:] - x[None,:,:], ord = 2, axis = 2, keepdims=False)
    elif len(x_shape) == 3:
        matrix = tf.norm(x[:,:,None,:] - x[:,None,:,:], ord = 2, axis = 3, keepdims=False)
    else:
        raise NotImplementedError
    return matrix

def compute_norm_pairwise_distance(x):
    ''' computation of normalized pairwise distance matrix
    ---- Input
    - x: input tensor		tf.Tensor [(bs), sample_num, dim_x]
    ---- Return
    - matrix: output matrix	tf.Tensor [(bs), sample_num, sample_num]
    '''
    x_pair_dist = compute_pairwise_distance(x)
    normalizer = tf.reduce_sum(x_pair_dist, axis=-1)
    normalizer = tf.stop_gradient(normalier + 1e-12)
    x_norm_pair_dist = tf.divide(x_pair_dist, normalizer)
    return x_norm_pair_dist

def NDiv_loss(z, y, alpha=0.8):
    ''' NDiv loss function.
    ---- Input
    - z: latent samples after embedding h_Z:		tf.Tensor [(bs), sample_num, dim_z].
    - y: corresponding outputs after embedding h_Y:	tf.Tensor [(bs), sample_num, dim_y].
    - alpha: hyperparameter alpha in NDiv loss.
    ---- Return
    - loss: normalized diversity loss.			tf.Tensor [(bs)]
    '''
    S = z.get_shape().as_list()[-2] # sample number
    y_norm_pair_dist = compute_norm_pairwise_distance(y)
    z_norm_pair_dist = compute_norm_pairwise_distance(z)
    ndiv_loss_matrix = tf.nn.relu(z_norm_pair_dist * alpha - y_norm_pair_dist)
    ndiv_loss = tf.reduce_sum(ndiv_loss_matrix, axis=(-2,-1)) / (S * (S - 1))
    return ndiv_loss



