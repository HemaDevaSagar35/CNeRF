import random
import torch.nn as nn
import torch
import time
from .fusion_aggregation import *


class Generator3d(nn.Module):
    """
    1. mostly inspired from https://github.com/MrTornado24/FENeRF/blob/main/generators/generators.py
    2. only forwad and stage forward are necessary.
    3. Also no need to hierarchical sampling, because CNeRF only uses 24 samples
    4. we need to initialize an extra sdf initializing function

    shapes for reference
    z_dim, hidden_dim, semantic_classes, blocks
    input, ray_directions, z_sample_one, z_sample_two = None

    TODO: Note:
    1. not implementing lock view dependence
    2. We probably need to test it later, after finishing discriminator and starting training loop
       By test I mean, sand some sample inputs and see how it responds and how the sizes are.
    """
    def __init__(self, generator_stack, z_dim, hidden_dim, latent_dim, semantic_classes, output_dim, blocks = 3, device = None):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.semantic_classes = semantic_classes
        self.blocks = blocks
        self.device = device
        self.output_dim = output_dim # 128 + 3 + 1 + 1 (useful during inference)

        self.generator_stack = generator_stack(
            z_dim = self.z_dim, 
            hidden_dim = self.hidden_dim,
            latent_dim = self.latent_dim,
            semantic_classes = self.semantic_classes,
            blocks = self.blocks)

        # apparently this could be learnable parameter.
        self.density_alpha = nn.Parameter(0.1*torch.ones(1))
    
    def residue_sdf(self, sdf, sdf_initial):
        # sdf : N x K x (imgximgx24) x (1)
        # sdf_initial : N x (imgximgx24) x 1
        #TODO : For now I am assuming the sphere is of radius 1. Maybe we need to check ano
        #another like stylesdf to see how sdf is calculated
        sdf_summed = sdf.sum(axis=-3)
        sdf_summed = sdf_summed + sdf_initial
        sdf_summed = -1.0*sdf_summed
        return torch.sigmoid(sdf_summed / self.density_alpha)/self.density_alpha

        #sigma : N x (imgximgx24) x 1
        # sigma = F.sigmoid(sdf_summed)
        # sigma = sigma /alpha

        # return sigma

    def forward(self, z_input_one, z_input_two, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, sample_dist=None, freq_bias_init = 30, 
                freq_std_init = 15, phase_bias_init = 0, phase_std_init = 0.25, **kwargs):
        
        ## Shapes:
        #### 1) transformed_points : n x rays x num_steps x 3
        #### 2) z_vals : n x rays x num_steps x 1
        #### 3) transformed_ray_directios : n x rays x 3
        #### 4) transformed_ray_origins : n x rays x 3
        #### 5) sdf_initial : n x rays x num_steps x 1 =>  n x (rays x num_steps) x 1 
        #### 6) pitch :  n x 1
        #### 7) yaw : n x 1

        batch_size = z_input_one.shape[0]
        latent_codes_combined = self.generator_stack.mix_latent_codes(z_input_one, z_input_two)

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam =  get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            #batch x (img x img x 24) x 1
            sdf_initial = self.generator.sdf_initial_values(transformed_points, transformed_ray_directions_expanded, 
                                latent_codes_combined, freq_bias_init, freq_std_init, phase_bias_init , phase_std_init)
            

        #coarse shape is N x K x (imgximgx24) x (128 + 3 + 1 + 1)
        # TODO: take special care about z_sample shapes
        coarse_output = self.generator_stack(transformed_points, transformed_ray_directions_expanded, latent_codes_combined, freq_bias_init, 
                                            freq_std_init, phase_bias_init, phase_std_init)
        

        # doing the semantic fusion and volume integration to get images and all
        #Note: 
        fused_frgb, sdf, mask = semantic_fusion(coarse_output)
        sigma = self.residue_sdf(sdf, sdf_initial)
        sigma = sigma.reshape((batch_size, img_size*img_size, n_steps, 1))
        #SHAPES NOTE:
        #frgb_final : n x (img_size*img_size) x (128 + 3)
        #mask_final : n x K x (img x img)
        # we shall handle the random picking of the semantic region in the training loop function
        frgb_final, mask_final = volume_aggregration(fused_frgb, sigma, mask, z_vals, batch_size, n_steps, img_size, semantic_classes = semantic_classes, noise_std=0.5)

        return frgb_final, torch.cat([pitch, yaw], axis=-1), mask_final
        

    def stage_forward(self, z_input_one, z_input_two, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, sample_dist=None, max_batch = 5, **kwargs):
        """Use this while inferencing.
        TODO: FNeRF, did some extra while inferencing, he took average of multiple freq and phase latent codes
        and used forward with freq and phase shifts in from siren, I didn't implement that
        phase shift in siren because I didn't know why we have to do that. I will incorporate if I find
        it useful later experiments.
        TODO: This should be reconstructed back again, we will do it in the end when the training pipeline is fully fixed.
        """
        ## Shapes:
        #### 1) transformed_points : n x rays x num_steps x 3
        #### 2) z_vals : n x rays x num_steps x 1
        #### 3) transformed_ray_directios : n x rays x 3
        #### 4) transformed_ray_origins : n x rays x 3
        #### 5) sdf_initial : n x rays x num_steps x 1 =>  n x (rays x num_steps) x 1 
        #### 6) pitch :  n x 1
        #### 7) yaw : n x 1

        if z_input_two is not None:
            raise("you gave another random sample")
        
        batch_size = z_input_one.shape[0]

        

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam =  get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, sdf_initial, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            sdf_initial = self.generator.sdf_initial_values(transformed_points, transformed_ray_directions_expanded, 
                                latent_codes_combined, freq_bias_init, freq_std_init, phase_bias_init , phase_std_init)

        #coarse shape is N x K x (imgximgx24) x (128 + 3 + 1 + 1)
        # TODO: take special care about z_sample shapes

        coarse_output = torch.zeros((batch_size, self.semantic_classes, img_size*img_size*num_steps, self.output_dim), device=self.device)
        quo = batch_size // max_batch
        rem = min(1, batch_size % max_batch)
        # TODO : we might face some issues for the loop below, haven't tested it yet. Need to do it
        # later after finishing discriminator


        for b in range(quo + rem):
            head = b*max_batch
            tail = (b+1)*max_batch

            coarse_output[head:tail,...] = self.generator_stack(
                transformed_points[head:tail,...], 
                transformed_ray_directions_expanded[head:tail,...], 
                z_sample_one = z_input_one[head:tail,...], 
                z_sample_two = None)
            
            # head += max_batch_size

        # doing the semantic fusion and volume integration to get images and all
        #Note: 
        fused_frgb, sdf, mask = fusion_aggregation(coarse_output)
        #SHAPES NOTE:
        #frgb_final : n x (img_size*img_size) x (128 + 3)
        #mask_final : n x K x (img x img)
        # we shall handle the random picking of the semantic region in the training loop function
        frgb_final, mask_final, sigma = volume_aggregration(fused_frgb, sdf, mask, sdf_initial, z_vals, n, n_steps, img_size, semantic_classes = semantic_classes, noise_std=0.5)

        pixels = frgb_final[...,-3:]
        pixels = pixels.reshape((batch_size, img_size, img_size, -1))

        mask_final = torch.nn.Softmax(dim=1)(mask_final)

        return pixels, mask_final, sigma









