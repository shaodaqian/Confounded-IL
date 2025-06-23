import os
import sys
import re
import six
import math
import torch
import pandas as pd

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
# import torchvision.transforms as transforms
import torch.nn.functional as F


# device='cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SimpleDataset(Dataset):
    def __init__(self,trainX,trainY,v1=None,v2=None):
        self.nSamples = len(trainX)
        # print('number of data: ',self.nSamples)
        self.trainX=trainX
        self.trainY=trainY
        self.v1=None
        self.v2=None
        if v1 is not None:
            self.v1=v1
        if v2 is not None:
            self.v2=v2

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if self.v1 is not None and self.v2 is not None:
            return (self.trainX[index],self.trainY[index],self.v1[index],self.v2[index])
        if self.v1 is not None:
            return (self.trainX[index],self.trainY[index],self.v1[index])
        else:
            return (self.trainX[index],self.trainY[index])


def log_norm_pdf(x, mu, log_sig):
    z = (x - mu) / (torch.exp(torch.clamp(log_sig, -40, 40))) #TODO: get rid of this clipping
    return -(0.5)*np.log(2*np.pi) - log_sig - 0.5*((z)**2)


def mixture_of_gaussian_loss(out,y_true):
    '''
    Combine the mixture of gaussian distribution and the loss into a single function
    so that we can do the log sum exp trick for numerical stability...
    '''
    [pi, mu, log_sig] = out
    # out shaped batch,dim,n_component
    gauss = log_norm_pdf(y_true.unsqueeze(-1).repeat(1,1,mu.shape[-1]),mu,log_sig)
    gauss = torch.clamp(gauss, -40, 40)
    max_gauss = max(0, torch.max(gauss))
    # log sum exp trick...
    gauss = gauss - max_gauss
    out = torch.sum(pi * torch.exp(gauss), dim=-1)
    loss = torch.mean(-torch.log(out) + max_gauss)
    return loss


def random_laplace(shape, mu=0., b=1.):
    '''
    Draw random samples from a Laplace distriubtion.

    See: https://en.wikipedia.org/wiki/Laplace_distribution#Generating_random_variables_according_to_the_Laplace_distribution
    '''
    # U = K.random_uniform(shape, -0.5, 0.5)
    U=torch.FloatTensor(shape).uniform_(-0.5, 0.5)
    return mu - b * torch.sign(U) * torch.log(1 - 2 * torch.abs(U))

def random_normal(mean=0.0, std=1.0):
    return torch.normal(mean, std)

def random_multinomial(logits, seed=None):
    '''
    Theano function for sampling from a multinomal with probability given by `logits`
    '''

    return F.one_hot(torch.multinomial(logits, num_samples=1),num_classes=int(logits.shape[-1])).squeeze()
    # return torch.multinomial(logits, num_samples=1)

def random_gmm(pi, mu, sig,n_compunents=5):
    '''
    Sample from a gaussian mixture model. Returns one sample for each row in
    the pi, mu and sig matrices... this is potentially wasteful (because you have to repeat
    the matrices n times if you want to get n samples), but makes it easy to implment
    code where the parameters vary as they are conditioned on different datapoints.
    '''
    # out shaped batch,dim,n_component

    normals = random_normal(mu, sig)

    # k = random_multinomial(pi)
    # normals=normals.reshape(normals.shape[0], n_compunents, -1)
    # k=k.reshape(k.shape[0], n_compunents, -1)
    # out=torch.sum(normals * k, dim=1)
    print(normals.shape,pi.shape)
    out=torch.sum(normals * pi, dim=-1)

    return out


def fit_linear(target: torch.Tensor,
               feature: torch.Tensor,
               reg: float = 0.0):
    """
    Parameters
    ----------
    target: torch.Tensor[nBatch, dim1, dim2, ...]
    feature: torch.Tensor[nBatch, feature_dim]
    reg: float
        value of l2 regularizer
    Returns
    -------
        weight: torch.Tensor[feature_dim, dim1, dim2, ...]
            weight of ridge linear regression. weight.size()[0] = feature_dim+1 if add_intercept is true
    """
    assert feature.dim() == 2
    assert target.dim() >= 2
    nData, nDim = feature.size()
    A = torch.matmul(feature.t(), feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device)
    # L = torch.linalg.cholesky(A)
    # A_inv = torch.cholesky_inverse(L)
    # TODO use cholesky version in the latest pytorch
    A_inv = torch.inverse(A)
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
        weight = torch.matmul(A_inv, b)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    return weight


def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor):
    assert weight.dim() >= 2
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)


def linear_reg_loss(target: torch.Tensor,
                    feature: torch.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    # print(torch.mean(torch.square(target - pred)),'mse')
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2


def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    """
    Parameters
    ----------
    mat1: torch.Tensor[nBatch, mat1_dim1, mat1_dim2, mat1_dim3, ...]
    mat2: torch.Tensor[nBatch, mat2_dim1, mat2_dim2, mat2_dim3, ...]

    Returns
    -------
    res : torch.Tensor[nBatch, mat1_dim1, ..., mat2_dim1, ...]
    """

    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2


def add_const_col(mat: torch.Tensor):
    """

    Parameters
    ----------
    mat : torch.Tensor[n_data, n_col]

    Returns
    -------
    res : torch.Tensor[n_data, n_col+1]
        add one column only contains 1.

    """
    assert mat.dim() == 2
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)
