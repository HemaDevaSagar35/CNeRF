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
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, n_blocks=3):
        super().__init__()
        self.network = [nn.Linear(z_dim, map_hidden_dim),
                        nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks):
            self.network.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))
        self.network = nn.Sequential(*self.network)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z) # z: (n_batch * n_point, n_channel)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts
    

class FiLMLayer(nn.Module):
    '''This layer follows the equation 
       phi(j) = sin((Wx + b)*freq + phase)
    '''
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        if x.shape[1] != freq.shape[1]:
            print("happening here")
            freq = freq.unsqueeze(1).expand_as(x) 
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)

class LocalGeneratorSiren(nn.Module):
    '''shape network is 3 layers deep and texture is 2 layers deep'''
    # Make sure the dimensions are consistent
    # summary(gen, [(1, 64*64*24, 3), (1, 1, 640),(1, 1, 640), (1,64*64*24, 3)], device = 'cpu')

    def __init__(self, hidden_dim=128, semantic_classes = 12):
        super().__init__()
        # self.device = device
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.output_dim = output_dim

        self.shape_network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim)
        ])

        self.texture_network = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim)
        ]) 

        self.feature_layer = nn.Linear(hidden_dim + 3, hidden_dim)
        self.color_layer = nn.Linear(hidden_dim, 3)
        self.mask_layer = nn.Linear(hidden_dim + 3, 1)
        self.sdf_layer = nn.Linear(hidden_dim, 1)

        # The initialization below are blatantly adapted from FeNERF repo. 
        #TODO: Their implication need to be further understood
        self.shape_network.apply(frequency_init(25))
        self.texture_network.apply(frequency_init(25))

        self.feature_layer.apply(frequency_init(25))
        self.color_layer.apply(frequency_init(25))
        self.mask_layer.apply(frequency_init(25))
        self.sdf_layer.apply(frequency_init(25))

        self.shape_network[0].apply(modified_first_sine_init)

        # self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, latent_code_freq, latent_code_phase, ray_directions):
        # ray direction is the view angles
        # print("input is")
        print(input.shape)
        input = self.gridwarper(input) #modifier 1
        x = input
        # latent_code_freq = latent_code_freq*15 + 30 #something we need to dabble with later
        # print(x.shape)
        for index, layer in enumerate(self.shape_network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, latent_code_freq[..., start:end], latent_code_phase[..., start:end])
        # print(x.shape)
        mask_input = torch.cat([ray_directions, x], axis=-1)

        for idx, layer in enumerate(self.texture_network):
            start = (index + idx + 1)*self.hidden_dim
            end = (index + idx + 2)*self.hidden_dim
            x = layer(x, latent_code_freq[..., start:end], latent_code_phase[..., start:end])
        
        feature_input = torch.cat([ray_directions, x], axis=-1)

        feature_output = self.feature_layer(feature_input)

        mask_output = self.mask_layer(mask_input)
        color_output = self.color_layer(feature_output)
        sdf_output = self.sdf_layer(x)

        return torch.cat([feature_output, color_output, mask_output, sdf_output], axis=-1)


class GeneratorStackSiren(nn.Module):
    '''Stack all the local generator networks
       hidden_dim=128, semantic_classes = 12, blocks = 3
    '''
    def __init__(self, z_dim, hidden_dim, semantic_classes, blocks):
        super().__init__()
        self.mapping_network = CustomMappingNetwork(z_dim, hidden_dim*2, hidden_dim*2*5, n_blocks=blocks)
        self.generator_list = []
        self.semantic_classes = semantic_classes
        for i in range(semantic_classes):
            self.generator_list.append(
                LocalGeneratorSiren(hidden_dim)
                )
        
        self.generator_list = nn.ModuleList(self.generator_list)
        
    def forward(self, input, ray_directions, z_sample_one, z_sample_two = None):
        # if sample two is not None then we sample between sample one and sample two latent codes
        # for every generator
        # output would batch size x K x (64x64 x 24) x (128 + 3 + 1 + 1)
        freq_sample_one, phase_sample_one = self.mapping_network(z_sample_one)
        freq_sample_two, phase_sample_two = self.mapping_network(z_sample_two) if z_sample_two is not None else (None, None)

        outputs = []
        for i in range(self.semantic_classes):
            print("happening for {}".format(i))
            sample = torch.randint(0, 2, (1,))[0]
            if (freq_sample_two is None) or (sample == 0):
                gen_output = self.generator_list[i](input, freq_sample_one, phase_sample_one, ray_directions)
                gen_output = torch.unsqueeze(gen_output, 1)
            else:
                gen_output = self.generator_list[i](input, freq_sample_two, phase_sample_two, ray_directions)
                gen_output = torch.unsqueeze(gen_output, 1)

            outputs.append(gen_output)
        
        return torch.cat(outputs, axis=1)

           






