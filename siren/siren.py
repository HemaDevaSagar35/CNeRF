# The following codes is adapted from 
#https://github.com/MrTornado24/FENeRF/blob/main/siren/siren.py

import sys
from numpy.lib.type_check import imag
# from torch._C import device

from torch.functional import align_tensors
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, n_blocks=3):
        super().__init__()
        self.network = [nn.Linear(z_dim, map_hidden_dim),
                        nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks-1):
            self.network.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
        
        # self.network.append(nn.Linear(map_hidden_dim, map_output_dim))
        self.network = nn.Sequential(*self.network)
        self.network.apply(kaiming_leaky_init)
        # with torch.no_grad():
        #     self.network[-1].weight *= 0.25

    def forward(self, z):
        # frequencies_offsets = self.network(z) # z: (n_batch * n_point, n_channel)
        # frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        # phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]
        latent_codes = self.network(z)
        # return frequencies, phase_shifts
        return latent_codes
    

class FiLMLayer(nn.Module):
    '''This layer follows the equation 
       phi(j) = sin((Wx + b)*freq + phase)
       assuming latent dim = 2*hidden_dim
    '''
    def __init__(self, input_dim, hidden_dim = 128, latent_dim = 256):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

        self.freq = nn.Linear(latent_dim, hidden_dim)
        self.phase = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x, latent_code, freq_bias_init = 30, freq_std_init = 15, phase_bias_init = 0, phase_std_init = 0.25):
        x = self.layer(x)
        # if x.shape[1] != freq.shape[1]:
        #     print("happening here")
        #     freq = freq.unsqueeze(1).expand_as(x) 
        #     phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        freq = self.freq(latent_code)
        phase_shift = self.phase(latent_code)
        return torch.sin(freq * x + phase_shift)

class LocalGeneratorSiren(nn.Module):
    '''shape network is 3 layers deep and texture is 2 layers deep
    parameters:
    1. (3 x 128 + 128 + 256*128*2) + 4*(128x128 + 128 + 256*128*2) = 512 + 65792 + 66048 + 263168 = 395520
    2. (131 x 128 + 128) + (128*3 + 3) + (131 + 1) + (128 + 1) = 16896 + 387 + 132 + 129 = 17544
    total = 395520 + 17544 = 413064

    '''
    # Make sure the dimensions are consistent
    # summary(gen, [(1, 64*64*24, 3), (1, 1, 640),(1, 1, 640), (1,64*64*24, 3)], device = 'cpu')

    def __init__(self, hidden_dim=128, latent_dim = 256, semantic_classes = 12):
        super().__init__()
        # self.device = device
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # # self.output_dim = output_dim

        self.shape_network = nn.ModuleList([
            FiLMLayer(3, hidden_dim, latent_dim),
            FiLMLayer(hidden_dim, hidden_dim,latent_dim),
            FiLMLayer(hidden_dim, hidden_dim, latent_dim)
        ])

        self.texture_network = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim, latent_dim),
            FiLMLayer(hidden_dim, hidden_dim, latent_dim)
        ]) 

        self.feature_layer = nn.Linear(hidden_dim + 3, hidden_dim)
        self.color_layer = nn.Linear(hidden_dim, 3)
        self.mask_layer = nn.Linear(hidden_dim + 3, 1)
        self.sdf_layer = nn.Linear(hidden_dim, 1)

        # # The initialization below are blatantly adapted from FeNERF repo. 
        # #: Their implication need to be further understood
        self.shape_network.apply(frequency_init(25))
        self.texture_network.apply(frequency_init(25))

        self.feature_layer.apply(frequency_init(25))
        self.color_layer.apply(frequency_init(25))
        self.mask_layer.apply(frequency_init(25))
        self.sdf_layer.apply(frequency_init(25))

        self.shape_network[0].apply(modified_first_sine_init)

        # # self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)
    
    # def get_init_sdf(self, input, latent_code, ray_directions, freq_bias_init = 30, freq_std_init = 15, phase_bias_init = 0, phase_std_init = 0.25):
    #     input = self.gridwarper(input) #modifier 1
    #     x = input
    #     j = 0
    #     for index, layer in enumerate(self.shape_network):
    #         x = layer(x, latent_code[0, j], freq_bias_init, freq_std_init, phase_bias_init, phase_std_init)
    #         j += 1
    #     for idx, layer in enumerate(self.texture_network):
    #         x = layer(x, latent_code[0, j], freq_bias_init, freq_std_init, phase_bias_init, phase_std_init)
    #         j += 1
    #     return self.sdf_layer(x)
    
    # def forward(self, x, latent_code):
        
    #     return self.shape_network(x, latent_code)

    def forward(self, input, latent_code, ray_directions, freq_bias_init = 30, freq_std_init = 15, phase_bias_init = 0, phase_std_init = 0.25):
        # ray direction is the view angles
        print("input is")
        print(input.shape)
        input = self.gridwarper(input) #modifier 1
        x = input
        # latent_code_freq = latent_code_freq*15 + 30 #something we need to dabble with later
        print(x.shape)
        j = 0
        for index, layer in enumerate(self.shape_network):  
            x = layer(x, latent_code[j], freq_bias_init, freq_std_init, phase_bias_init, phase_std_init)
            j += 1
        
        mask_input = torch.cat([ray_directions, x], axis=-1)

        for idx, layer in enumerate(self.texture_network):
            x = layer(x, latent_code[j], freq_bias_init, freq_std_init, phase_bias_init, phase_std_init)
            j += 1

        feature_input = torch.cat([ray_directions, x], axis=-1)

        feature_output = self.feature_layer(feature_input)

        mask_output = self.mask_layer(mask_input)
        color_output = self.color_layer(feature_output)
        sdf_output = self.sdf_layer(x)
      
        return torch.cat([feature_output, color_output, mask_output, sdf_output], axis=-1)



class GeneratorStackSiren(nn.Module):
    '''Stack all the local generator networks
       hidden_dim=128, semantic_classes = 12, blocks = 3

       parameters calculation: 
        1. mapping: (100*256 + 256) + 2*(256*256 + 256) = 25856 + 131584 = 157440
        2. generator: 12*413064 = 4956768
        total = 157440 + 4956768 = 5114208
    '''
    def __init__(self, z_dim, hidden_dim, latent_dim, semantic_classes, blocks):
        super().__init__()
        # self.device = device
        self.mapping_network = CustomMappingNetwork(z_dim, latent_dim, n_blocks=blocks)
        self.generator_list = []
        self.semantic_classes = semantic_classes
        for i in range(semantic_classes):
            self.generator_list.append(
                LocalGeneratorSiren(hidden_dim, latent_dim, semantic_classes)
                )
        
        self.generator_list = nn.ModuleList(self.generator_list)

        self.sdf_init_network = LocalGeneratorSiren(hidden_dim, latent_dim, semantic_classes)

    def extract_latent(self, input):
        return self.mapping_network(input)
    
    

    def shuffle_latent(self, semantic_classes, latent_code_one, latent_code_two, shape_depth=3, texture_depth = 2):
        # I am only shuffling full latent code not doing shuffling at texture and shape level
        # N x 256 , N x 256 - 12
        #output shape is 12 x 5 x N x 256
        
        shuffled_codes = []
        for i in range(semantic_classes):
            ith_level = []
            for j in range(latent_code_one.shape[0]):
                sample = torch.randint(0, 3, (1,))[0]
                if sample == 0:
                    ith_level.append(latent_code_one[j:j+1,:].repeat(shape_depth+texture_depth, 1).unsqueeze(1))
                elif sample == 2:
                    ith_level.append(latent_code_two[j:j+1,:].repeat(shape_depth+texture_depth, 1).unsqueeze(1))
                else:
                    ith_level.append(torch.cat([latent_code_one[j:j+1,:].repeat(shape_depth, 1),
                        latent_code_two[j:j+1,:].repeat(texture_depth, 1)],axis=0).unsqueeze(1))
            #5 x N x 256
            ith_level = torch.cat(ith_level, axis=1)
            ith_level = ith_level.unsqueeze(0)

            shuffled_codes.append(ith_level)
        return torch.cat(shuffled_codes, axis=0)
    
    def mix_latent_codes(self, z_sample_one, z_sample_two):
        #output shape 12 x 5 x N x 256
        if z_sample_two is None:
            latent_code_combined = self.extract_latent(torch.cat([z_sample_one, z_sample_one], axis=0))
        else:
            latent_code_combined = self.extract_latent(torch.cat([z_sample_one, z_sample_two], axis=0))
        N = latent_code_combined.shape[0] // 2
        latent_codes_combined = self.shuffle_latent(self.semantic_classes, latent_code_combined[:N,:], latent_code_combined[N:,:])
        return latent_codes_combined

    def sdf_initial_values(self, input, ray_directions, latent_codes_combined, freq_bias_init = 30, freq_std_init = 15, phase_bias_init = 0, phase_std_init = 0.25):

        return self.sdf_init_network(input, latent_codes_combined[0], ray_directions, freq_bias_init,  freq_std_init, phase_bias_init, phase_std_init)

    def forward(self, input, ray_directions, latent_codes_combined, freq_bias_init = 30, freq_std_init = 15, phase_bias_init = 0, phase_std_init = 0.25):
        #Note this is for only training.
        # if sample two is not None then we sample between sample one and sample two latent codes
        # for every generator
        # output would batch size x K x (64x64 x 24) x (128 + 3 + 1 + 1)
        
        outputs = []
        for i in range(self.semantic_classes):
            gen_output = self.generator_list[i](input, latent_codes_combined[i], ray_directions, freq_bias_init,  freq_std_init, phase_bias_init, phase_std_init)
            gen_output = torch.unsqueeze(gen_output, 1)
        
            outputs.append(gen_output)
        
        return torch.cat(outputs, axis=1)

    






