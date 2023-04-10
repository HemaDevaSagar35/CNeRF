
# inspired from https://github.com/MrTornado24/FENeRF/blob/9d90acda243b7c7d7f2c688a3bb333da2e7f8894/discriminators/discriminators.py#L21

import math
import torch
import torch.nn as nn
import sys

# import curriculums
import numpy as np
import torch.nn.functional as F


#NOTE: Implementing residualcoordconv. This is not specifically mentioned in CNeRF,
# but abstracting it from FeNeRF

class AdapterBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2)
        )
    def forward(self, input):
        return self.model(input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

class ResidualCoordConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=False, groups=1):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
        self.downsample = downsample

    def forward(self, identity):
        y = self.network(identity)

        if self.downsample: y = nn.functional.avg_pool2d(y, 2)
        if self.downsample: identity = nn.functional.avg_pool2d(identity, 2)
        identity = identity if self.proj is None else self.proj(identity)

        y = (y + identity)/math.sqrt(2)
        return y


class GlobalDiscriminator(nn.Module):
    """
    This is global discriminator. and we predicted real/fake along with view direction 
    angle
    NOTE: we are not doing progressive training here. Do framing with nn.sequential
    NOTE: I am summing the ouptut from both branches. Not clear from the paper
    if it should be sum or append. But for now I am going to sum.
    """
    def __init__(self, semantic_classes=12):
        super().__init__()
        self.color_layers = nn.Sequential( 
            AdapterBlock(3, 16),
            ResidualCoordConvBlock(16, 32, downsample=True), # 64 x 64 --> 32 x 32
            ResidualCoordConvBlock(32, 64, downsample=True), # 32 x 32 ---> 16 x 16
            ResidualCoordConvBlock(64, 128, downsample=True), # 16 x 16 ---> 8 x 8
            ResidualCoordConvBlock(128, 256, downsample=True), # 8 x 8 ---> 4 x 4
            ResidualCoordConvBlock(256, 400, downsample=True), # 4 x 4 ---> 2 x 2
        
        )

        self.mask_layers = nn.Sequential( 
            AdapterBlock(semantic_classes, 16),
            ResidualCoordConvBlock(16, 32, downsample=True), # 64 x 64 --> 32 x 32
            ResidualCoordConvBlock(32, 64, downsample=True), # 32 x 32 ---> 16 x 16
            ResidualCoordConvBlock(64, 128, downsample=True), # 16 x 16 ---> 8 x 8
            ResidualCoordConvBlock(128, 256, downsample=True), # 8 x 8 ---> 4 x 4
            ResidualCoordConvBlock(256, 400, downsample=True), # 4 x 4 ---> 2 x 2
        
        )

        self.final_layer = nn.Conv2d(400, 3, 2)
    
    def forward(self, color_input, mask_input):
        color_output = self.color_layers(color_input) 
        print("issue is here")
        mask_output = self.mask_layers(mask_input)

        x = color_output + mask_output
        x = self.final_layer(x).reshape(x.shape[0], 3)
        
        return x[...,:1], x[...,1:]

class SemanticDiscriminator(nn.Module):
    """
    This is semantic discriminator. Output would be 1 + 12(semantic labels)
    output for final layer is not given in the paper. For now keeping at
    256
    GD --
    """
    def __init__(self, semantic_classes=12):
        super().__init__()
        self.color_layers = nn.Sequential( 
            AdapterBlock(3, 16),
            ResidualCoordConvBlock(16, 32, downsample=True), # 64 x 64 --> 32 x 32
            ResidualCoordConvBlock(32, 64, downsample=True), # 32 x 32 ---> 16 x 16
            ResidualCoordConvBlock(64, 128, downsample=True), # 16 x 16 ---> 8 x 8
            ResidualCoordConvBlock(128, 256, downsample=True), # 8 x 8 ---> 4 x 4
            ResidualCoordConvBlock(256, 400, downsample=True), # 4 x 4 ---> 2 x 2
        
        )

        self.final_layer = nn.Conv2d(400, 256, 2)

        self.fc_label = nn.Linear(256, 1)
        self.mask_label = nn.Linear(256, semantic_classes)
    
    def forward(self, input):
        x = self.color_layers(input)
        x = self.final_layer(x)

        x = torch.flatten(x, 1)
        return self.fc_label(x), self.mask_label(x)

