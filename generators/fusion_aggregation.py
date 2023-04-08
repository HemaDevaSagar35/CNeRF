
import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def semantic_fusion(output_generator):
    # output_generator shape : N x K x (imgximgx24) x (128 + 3 + 1 + 1)
    frgb = output_generator[...,:-2]
    mask = output_generator[...,-2:-1]
    sdf = output_generator[...,-1:]

    fused_frgb = frgb * mask
    # N x (imgximgx24) x (128 + 3)
    fused_frgb = fused_frgb.sum(axis=-3)
    return fused_frgb, sdf, mask

def residue_sdf(sdf, sdf_initial, alpha = 1.0):
    # sdf : N x K x (imgximgx24) x (1)
    # sdf_initial : N x (imgximgx24) x 1
    #TODO : For now I am assuming the sphere is of radius 1. Maybe we need to check ano
    #another like stylesdf to see how sdf is calculated
    sdf_summed = sdf.sum(axis=-3)
    sdf_summed = sdf_summed + sdf_initial
    sdf_summed = -1.0*sdf_summed
    sdf_summed = sdf_summed / alpha

    #sigma : N x (imgximgx24) x 1
    sigma = F.sigmoid(sdf_summed)
    sigma = sigma /alpha

    return sigma
    

def volume_aggregration(fused_frgb, sdf, mask, sdf_initial, z_vals, n, n_steps, img_size, semantic_classes = 12, noise_std=0.5):
    #fused_frgb : N x (imgximgx24) x (128 + 3)
    # sdf : N x K x (imgximgx24) x (1)
    # sdf_initial : N x (imgximgx24) x 1
    # z_vals : N x (img x img) x 24 x 1
    # n is batch_size

    #TODO: SOme variations are there for this from the FNeRF's fancy intergration
    # in terms of back fill and all
    #TODO: need to take care about device
    sigma = residue_sdf(sdf, sdf_initial)

    # re-shape sigma and fused_frgb to N x (img x img) x 24 x (128 + 3)
    fused_frgb = fused_frgb.reshape((n, img_size*img_size, n_steps, fused_frgb.shape[-1]))
    sigma = sigma.reshape((n, img_size*img_size, n_steps, 1))

    # volume rendering equation
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigma.shape) * noise_std
    alphas = 1 - torch.exp(-deltas * (F.relu(sigma + noise)))
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    # weights : N x (img x img) x 24 x 1
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]

    #frgb_final : n x (img_size*img_size) x (128 + 3)
    frgb_final = torch.sum(fused_frgb*weights, axis=-2)

    #mask
    
    weights = (weights[...,-1]).unsqueeze(-3).repeat(1, semantic_classes, 1, 1)
    mask = mask[...,-1].reshape((n, semantic_classes, img_size*img_size, n_steps))
    #mask_final : N x K x (img x img)
 
    mask_final = torch.sum(mask*weights, axis=-1)
    return frgb_final, mask_final

    # test case to see if this is working fine or not


if __name__ == "__main__":
    #below is the test case for volume aggregation
    n = 2
    n_steps = 2
    img_size = 2
    semantic_classes = 2
    noise_std=0.5

    #matrices
    #N x (imgximgx24) x (128 + 3) # 5
    fused_frgb = torch.tensor([[0.5, 0.75, 0.2, 0.67, 0.85],
    [0.75, 0.2,0.5,0.33, 0.59],
    [0.5, 0.75, 0.2, 0.33, 0.59],
    [ 0.67, 0.85, 0.2,0.5,0.33],
    [0.75, 0.2,0.5,0.33, 0.59],
    [ 0.67, 0.85, 0.2,0.5,0.33],
    [0.5, 0.75, 0.2, 0.67, 0.85],
    [0.75, 0.2,0.5,0.33, 0.59],
    ])
    fused_frgb = fused_frgb.unsqueeze(0).repeat(n, 1, 1)
    # sdf : N x K x (imgximgx24) x (1)

    sdf = torch.tensor([[[0.5],
    [0.75],
    [0.5],
    [ 0.67],
    [0.67],
    [0.85],
    [0.5],
    [0.33]
    ],[
     [0.15],
    [0.35],
    [ 0.27],
    [0.22],
    [0.95],
    [0.15],
    [0.63],
    [0.58]  
    ]
    ]
    )
    sdf = sdf.unsqueeze(0).repeat(n, 1, 1, 1)

    #mask : N x K x (imgximgx24) x (1)
    mask =  torch.tensor([[[0.25],
    [0.55],
    [0.5],
    [ 0.67],
    [0.77],
    [0.85],
    [0.35],
    [0.23]
    ],[
     [0.85],
    [0.95],
    [ 0.17],
    [0.12],
    [0.75],
    [0.15],
    [0.23],[0.66]   
    ]
    ]
    )

    mask = mask.unsqueeze(0).repeat(n, 1, 1, 1)

    #sdf_initial  # sdf : N x (imgximgx24) x (1)
    sdf_initial = torch.tensor([[0.5],
    [0.5],
    [0.5],
    [ 0.7],
    [0.7],
    [0.5],
    [0.5],
    [0.3]
    ])

    sdf_initial = sdf_initial.unsqueeze(0).repeat(n, 1, 1)

    #z_vals : N x (img x img) x 24 x 1
    z_vals = torch.tensor([[[0.5],
    [0.5],
    ],[
    [0.5],
    [ 0.7]
    ],[
    [0.7],
    [0.9]],
    [
    [0.5],
    [0.65]]
    ])
    z_vals = z_vals.unsqueeze(0).repeat(n, 1, 1, 1)

    print("fused rgb is ",fused_frgb.shape)
    print("sdf is ", sdf.shape)
    print("mask is ", mask.shape)
    print("sdf initial is ",sdf_initial.shape)
    print("z vals is ", z_vals.shape)

    frgb_final, mask_final = volume_aggregration(fused_frgb, sdf, mask, sdf_initial, z_vals, n, n_steps, img_size, semantic_classes = semantic_classes, noise_std=0.5)
    print("frgb ",frgb_final.shape)
    print("mask is ", mask_final.shape)
    print(frgb_final)
    print(mask_final)









