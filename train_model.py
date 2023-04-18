import argparse
import os
import numpy as np
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.summary import make_image
from torchvision.utils import save_image, make_grid

from generators import generator
from discriminators import discriminator
from losses import *

from siren import siren
import fid_evaluation #need to copy this script from FeNeRF

import datasets #need to implement this
import curriculums
from tqdm import tqdm

import copy
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter

COLOR_MAP = {
            0: [0, 0, 0], 
            1: [204, 0, 0],
            2: [76, 153, 0], 
            3: [204, 204, 0], 
            4: [51, 51, 255], 
            5: [204, 0, 204], 
            6: [0, 255, 255], 
            7: [255, 204, 204], 
            8: [102, 51, 0], 
            9: [255, 0, 0], 
            10: [102, 204, 0], 
            11: [255, 255, 0], 
            12: [0, 0, 153], 
            13: [0, 0, 204], 
            14: [255, 51, 153], 
            15: [0, 204, 204], 
            16: [0, 51, 0], 
            17: [255, 153, 51], 
            18: [0, 204, 0]}

def mask2color(masks):
    masks = torch.argmax(masks, dim=1).float()
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float)
    for key in COLOR_MAP:
        sample_mask[masks==key] = torch.tensor(COLOR_MAP[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z

def train(rank, world_size, opt):
    """
    opt is the argument dict it has
     1. port
     2. output_dir
     3. load_dir
     4. load_step

    """
    torch.manual_seed(0)
    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    #TODO: need to implement curriculums and modify it
    metadata = curriculums.extract_metadata(curriculum, 0)

    #curriculum dict
    z_sample_fixed = z_sampler((25, metadata['z_dim']), device='cpu', dist=metadata['z_dist'])
    CHANNELS = 3
    #curriculum dict
    CHANNELS_SEG = curriculum.get('channel_seg', 18)

    scaler = torch.cuda.amp.GradScaler()

    if rank == 0:
        logger = SummaryWriter(os.path.join(opt.output_dir, 'logs'))

    if opt.load_dir != '':
        if opt.load_step == 0:
            generator_all = torch.load(os.path.join(opt.load_dir, 'generator_all.pth'), map_location=device)
            discriminator_global = torch.load(os.path.join(opt.load_dir, 'discriminator_global.pth'), map_location=device)
            discriminator_local = torch.load(os.path.join(opt.load_dir, 'discriminator_local.pth'), map_location=device)
            ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
            ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
        else:
            generator_all = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_generator_all.pth'), map_location=device)
            discriminator_global = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_discriminator_global.pth'), map_location=device)
            discriminator_local = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_discriminator_local.pth'), map_location=device)
            ema = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_ema.pth'), map_location=device)
            ema2 = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_ema2.pth'), map_location=device)
    else:
        generator_all = generator.Generator3d(siren.GeneratorStackSiren, 
            z_dim = metadata['z_dim'],
            hidden_dim = metadata['hidden_dim'],
            latent_dim = metadata['latent_dim'],
            semantic_classes = metadata['semantic_classes'],
            output_dim = metadata['output_dim'], 
            blocks = metadata['blocks'])
    

        discriminator_global = discriminator.GlobalDiscriminator(
            metadata['semantic_classes']
        )

        discriminator_local = discriminator.SemanticDiscriminator(
            metadata['semantic_classes']
        )

        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    generator_ddp = DDP(generator_all, device_ids = [rank], find_unused_parameters=True)
    discriminator_global_ddp = DDP(discriminator_global, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    discriminator_local_ddp = DDP(discriminator_local, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)

    #need to understand this step further, why is getting re-assigned
    generator_all = generator_ddp.module
    discriminator_global = discriminator_global_ddp.module
    discriminator_local = discriminator_local_ddp.module

    optimizer_G = torch.optim.Adam(generator_ddp.parameters(),
        lr = metadata['gen_lr'],
        betas = metadata['betas'],
        weight_decay = metadata['weight_decay']
    )

    optimizer_global_D = torch.optim.Adam(
        discriminator_global_ddp.parameters(),
        lr = metadata['gen_d'],
        betas = metadata['betas'],
        weight_decay = metadata['weight_decay']
    )

    optimizer_local_D = torch.optim.Adam(
        discriminator_local_ddp.parameters(),
        lr = metadata['gen_d'],
        betas = metadata['betas'],
        weight_decay = metadata['weight_decay']
    )

    generator_losses = []
    discriminator_losses = [] #we combine both local and segment

    if opt.set_step != None:
        generator_all.step = opt.set_step
        discriminator_global.step = opt.set_step
        discriminator_local.step = opt.set_step
    

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    generator_all.device = device #caution: need to be careful wi

    #------
    # Training stuff
    #------
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator_all))
        f.write('\n\n')
        f.write(str(discriminator_global))
        f.write(str(discriminator_global))
        f.write('\n\n')
        f.write(str(curriculum))
    
    torch.manual_seed(rank)
    dataloader, CHANNELS = datasets.get_dataset_distributed(
        metadata['dataset'],
        world_size,
        rank,
        **metadata
    )

    total_progress_bar = tqdm(
        total = opt.n_epochs, desc = 'Total progress',
        dynamic_ncols = True
    )

    total_progress_bar.update(discriminator_global.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)
        for i, (imgs, label, _) in enumerate(dataloader):
            if discriminator_global.step % opt.model_save_interval == 0 and rank == 0:
                #save stuff
                torch.save(ema, os.path.join(opt.output_dir, str(discriminator_global.step) + '_ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, str(discriminator_global.step) + '_ema2.pth'))
                torch.save(generator_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_generator.pth'))
                torch.save(discriminator_global_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_discriminator_global.pth'))
                torch.save(discriminator_local_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_discriminator_local.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_optimizer_G.pth'))
                torch.save(optimizer_global_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_optimizer_global_D.pth'))
                torch.save(optimizer_local_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_optimizer_local_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_scaler.pth'))

            if scaler.get_scale() < 1:
                scaler.update(1.)
            
            generator_ddp.train()
            discriminator_global_ddp.train()
            discriminator_local_ddp.train()

            real_imgs = imgs.to(device, non_blocking=True).float()
            real_labels = label.to(device, non_blocking=True).float()
            #Caution
            #metadata['nerf_noise'] = max(0, 1 - discriminator_global.step/5000.)

            #Train Global Discriminator
            with torch.cuda.amp.autocast():
                #first generate images for discriminator training
                with torch.no_grad():
                    z_sample_one = z_sampler((real_imgs.shape[0], metadata['z_dim']), device=device, dist=metadata['z_dist'])
                    z_sample_two = z_sampler((real_imgs.shape[0], metadata['z_dim']), device=device, dist=metadata['z_dist'])

                    split_batch_size = z_sample_one.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    gen_masks = []

                    for split in range(metadata['batch_split']):
                        z_sample_one_subset = z_sample_one[split * split_batch_size:(split+1) * split_batch_size]
                        z_sample_two_subset = z_sample_two[split * split_batch_size:(split+1) * split_batch_size]

                        #TODO: metadata should contain image size too
                        g_imgs, g_pos, g_mask = generator_ddp(
                            z_sample_one_subset,
                            z_sample_two_subset,
                            **metadata)
                        
                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)
                        gen_masks.append(g_mask)

                    
                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)
                    gen_masks = torch.cat(gen_masks, axis=0)

                real_imgs.requires_grad = True
                real_labels.requires_grad = True #Losses
                #Losses: real labels here as masks and they should have same shape as gen_masks
                #Losses: we sent both image and mask as inputs to the discriminator
                r_img_preds, _ = discriminator_global_ddp(real_imgs, real_labels)

            #LOSS: we need to check the r1 lambda values
            grad_img_penalty, grad_mask_penalty = discriminator_loss_r1(r_img_preds, real_imgs, real_labels, scaler)
            
            #Curriculum
            with torch.cuda.amp.autocast():
                grad_r1_penalty = 0.5*metadata['r1_img_lambda']*grad_img_penalty + 0.5*metadata['r1_mask_lambda']*grad_mask_penalty
                g_img_preds, g_img_positions_pred = discriminator_global_ddp(gen_imgs, gen_masks)

                g_position_loss = torch.nn.SmoothL1Loss()(g_img_positions_pred, gen_positions) * metadata['pos_lambda']
                d_global_loss = torch.nn.functional.softplus(g_img_preds).mean() + torch.nn.functional.softplus(-r_img_preds).mean() + \
                    grad_r1_penalty + g_position_loss
                
                discriminator_losses.append(d_global_loss.item())
            
            if rank == 0:
                logger.add_scalar('d_global_loss', d_global_loss.item(), discriminator_global.step)
            
            optimizer_global_D.zero_grad()
            scaler.scale(d_global_loss).backward()
            scaler.unscale_(optimizer_global_D)
            ##Caution for grad clipping.
            torch.nn.utils.clip_grad_norm_(discriminator_global_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_global_D)

            #Train local discriminator
            

            
            


            





                        




