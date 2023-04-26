
import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from .math_utils_torch import *

def semantic_fusion(output_generator):
    # output_generator shape : N x K x (imgximgx24) x (128 + 3 + 1 + 1)
    frgb = output_generator[...,:-2]
    mask = output_generator[...,-2:-1]
    sdf = output_generator[...,-1:]
    print("Before fusing rgb max is, ", torch.max(frgb[...,-3:]))
    print("Before fusing rgb min is, ", torch.min(frgb[...,-3:]))
    fused_frgb = frgb * mask
    print("After fusing rgb max is, ", torch.max(fused_frgb[...,-3:]))
    print("After fusing rgb min is, ", torch.min(fused_frgb[...,-3:]))
    # N x (imgximgx24) x (128 + 3)
    fused_frgb = fused_frgb.sum(axis=-3)
    print("After fusing rgb summing max is, ", torch.max(fused_frgb[...,-3:]))
    print("After fusing rgb summing min is, ", torch.min(fused_frgb[...,-3:]))
    return fused_frgb, sdf, mask


    

def volume_aggregration(fused_frgb, sigma, mask, z_vals, n, n_steps, img_size, device, semantic_classes = 12, noise_std=0.5):
    #fused_frgb : N x (imgximgx24) x (128 + 3)
    # sdf : N x K x (imgximgx24) x (1)
    # sdf_initial : N x (imgximgx24) x 1
    # z_vals : N x (img x img) x 24 x 1
    # n is batch_size
    # mask :  N x K x (imgximgx24) x (1)

    #TODO: SOme variations are there for this from the FNeRF's fancy intergration
    # in terms of back fill and all
    #TODO: need to take care about device
    # sigma = residue_sdf(sdf, sdf_initial)

    # re-shape sigma and fused_frgb to N x (img x img) x 24 x (128 + 3)
    fused_frgb = fused_frgb.reshape((n, img_size*img_size, n_steps, fused_frgb.shape[-1]))
    # sigma = sigma.reshape((n, img_size*img_size, n_steps, 1))
    # print("REACHED 1")
    # volume rendering equation
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)
    # print("REACHED 2")
    noise = torch.randn(sigma.shape, device=device) * noise_std
    alphas = 1 - torch.exp(-deltas * (F.relu(sigma + noise)))
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    # weights : N x (img x img) x 24 x 1
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]

    #frgb_final : n x (img_size*img_size) x (128 + 3)
    print("checking max no change, ", torch.max(fused_frgb[...,-3:]))
    print("checking min no change, ", torch.min(fused_frgb[...,-3:]))
    frgb_final = torch.sum(fused_frgb*weights, axis=-2)
    print("after w multi max no change, ", torch.max(frgb_final[...,-3:]))
    print("after w multi min no change, ", torch.min(frgb_final[...,-3:]))

    print("w multi max , ", torch.max(weights))
    print("w multi min , ", torch.min(weights))

    frgb_final[:,:,-3:] = -1 + 2*frgb_final[:,:,-3:]
    print("after normalize w multi max no change, ", torch.max(frgb_final[...,-3:]))
    print("after normalize w multi min no change, ", torch.min(frgb_final[...,-3:]))
    # print("REACHED 3")
    #mask
    
    weights = (weights[...,-1]).unsqueeze(-3).repeat(1, semantic_classes, 1, 1)
    mask = mask[...,-1].reshape((n, semantic_classes, img_size*img_size, n_steps))
    #mask_final : N x K x img x img
 
    mask_final = torch.sum(mask*weights, axis=-1)
    print("before final pass max no change, ", torch.max(frgb_final[...,-3:]))
    print("before final pass min no change, ", torch.min(frgb_final[...,-3:]))
    return frgb_final.transpose(2, 1).reshape((n, -1, img_size, img_size)), mask_final.reshape((n, semantic_classes, img_size, img_size))

    # test case to see if this is working fine or not


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))


    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals

def transform_sampled_points(points, z_vals, ray_directions, device, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal'):
    """Samples a camera position and maps points in camera space to world space."""
    # New addition: sdf_initial
    #### 1) for now assuming sdf for sphere as sqrt(x**2 + y**2 + z**2) - 1 (where radius = 1)

    n, num_rays, num_steps, channels = points.shape

    points, z_vals = perturb_points(points, z_vals, ray_directions, device)


    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)


    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    # sdf_intial = sdf_initial_calculate(transformed_points[..., :3], device)

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw

def sdf_initial_calculate(points, device):
    # points : n x num_rays x num_steps x channel
    #TODO: Haven't test this function. Maybe we have to do it at later point
    sdf_initial = torch.norm(points, dim=-1, keepdim=True)
    r_values = torch.ones_like(sdf_initial, device=device)
    return sdf_initial - r_values


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    n: batch size
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    randn: N(0, 1)
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    # world coordinate is the same as camera
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta) # x axis
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta) # z axis
    output_points[:, 1:2] = r*torch.cos(phi) # y axis

    return output_points, phi, theta

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world



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









