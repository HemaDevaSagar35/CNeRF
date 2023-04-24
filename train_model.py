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

def sum_params(model):
    s = 0.0
    for p in model.parameters():
        s += p.cpu().data.numpy().sum()
    return s

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
    # CHANNELS_SEG = curriculum.get('channel_seg', 12)

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
            output_dim = metadata['output_dim'], scalar = scaler, 
            blocks = metadata['blocks']).to(device)
    

        discriminator_global = discriminator.GlobalDiscriminator(
            metadata['semantic_classes']
        ).to(device)

        discriminator_local = discriminator.SemanticDiscriminator(
            metadata['semantic_classes']
        ).to(device)

        ema = ExponentialMovingAverage(generator_all.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator_all.parameters(), decay=0.9999)

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
                torch.save(ema.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_ema.pth'))
                torch.save(ema2.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_ema2.pth'))
                torch.save(generator_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_generator_all.pth'))
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
                    flip = torch.rand(1).to(device)
                    if flip[0] <= metadata['mixing_bool']:
                        z_sample_two = z_sampler((real_imgs.shape[0], metadata['z_dim']), device=device, dist=metadata['z_dist'])
                    else:
                        z_sample_two=None

                    split_batch_size = z_sample_one.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    gen_masks = []

                    for split in range(metadata['batch_split']):
                        z_sample_one_subset = z_sample_one[split * split_batch_size:(split+1) * split_batch_size]
                        z_sample_two_subset = None if z_sample_two is None else z_sample_two[split * split_batch_size:(split+1) * split_batch_size]

                        #TODO: metadata should contain image size too
                        g_imgs, g_pos, g_mask, _, _ = generator_ddp(
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
                g_img_preds, g_img_positions_pred = discriminator_global_ddp(gen_imgs.detach()[:,-3:], gen_masks.detach())

                g_position_loss = torch.nn.SmoothL1Loss()(g_img_positions_pred, gen_positions.detach()) * metadata['pos_lambda']
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
            #randomely choose which semantic mask to try for
            semantic_choices = torch.randint(0, metadata['semantic_classes'], (real_imgs.shape[0],)).to(device)
            gen_mask_choice = torch.zeros((real_imgs.shape[0], 1, real_imgs.shape[2], real_imgs.shape[3])).to(device)
            real_mask_choice = torch.zeros((real_imgs.shape[0], 1, real_imgs.shape[2], real_imgs.shape[3])).to(device)
            for j, sem_label in enumerate(semantic_choices):
                gen_mask_choice[j, 0] = gen_masks[j, sem_label]
                real_mask_choice[j, 0] = real_labels[j, sem_label]

            with torch.cuda.amp.autocast():
                r_sem_img_preds, r_sem_mask_preds = discriminator_local_ddp(real_imgs, real_mask_choice)
            
            # real_mask_choice.requires_grad = True
            grad_seg_img_penalty, grad_seg_mask_penalty = discriminator_loss_r1(r_sem_img_preds, real_imgs, real_mask_choice, scaler)
            with torch.cuda.amp.autocast():
                grad_r1_seg_penalty = 0.5*grad_seg_img_penalty*metadata['r1_img_lambda'] + 0.5*grad_seg_mask_penalty*metadata['r1_mask_lambda']
                g_sem_img_preds, g_sem_mask_preds = discriminator_local_ddp(gen_imgs.detach()[:,-3:], gen_mask_choice.detach())

                g_sem_cross_entropy = torch.nn.CrossEntropyLoss()(g_sem_mask_preds, semantic_choices)
                r_sem_cross_entropy = torch.nn.CrossEntropyLoss()(r_sem_mask_preds, semantic_choices)

                d_local_loss = (torch.nn.functional.softplus(-r_sem_img_preds).mean() + \
                               torch.nn.functional.softplus(g_sem_img_preds).mean() + \
                               grad_r1_seg_penalty + g_sem_cross_entropy + r_sem_cross_entropy)*metadata['local_d_lambda']

            
            if rank == 0:
                logger.add_scalar('d_local_loss', d_local_loss.item(), discriminator_local.step)
            
            optimizer_local_D.zero_grad()
            scaler.scale(d_local_loss).backward()
            scaler.unscale_(optimizer_local_D)
            torch.nn.utils.clip_grad_norm_(discriminator_local_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_local_D)

            discriminator_losses.append(d_local_loss.item())
            total_d_loss = d_global_loss.detach().item() + d_local_loss.detach().item()
            if rank == 0:
                logger.add_scalar("d_loss (global + local)", total_d_loss, discriminator_local.step)
            

            #Train the generator
            z_sample_one = z_sampler((real_imgs.shape[0], metadata['z_dim']), device=device, dist=metadata['z_dist'])
            flip = torch.rand(1).to(device)
            if flip[0] <= metadata['mixing_bool']:
                z_sample_two = z_sampler((real_imgs.shape[0], metadata['z_dim']), device=device, dist=metadata['z_dist'])
            else:
                z_sample_two=None
            
            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    z_sample_one_subset = z_sample_one[split * split_batch_size:(split+1) * split_batch_size]
                    z_sample_two_subset = None if z_sample_two is None else z_sample_two[split * split_batch_size:(split+1) * split_batch_size]
                    g_imgs, g_pos, g_mask, g_grad_sdf, g_sdf = generator_ddp(z_sample_one_subset, z_sample_two_subset, **metadata)

                    semantic_choices = torch.randint(0, metadata['semantic_classes'], (z_sample_one_subset.shape[0],)).to(device)
                    g_mask_choice = torch.zeros((z_sample_one_subset.shape[0], 1, real_imgs.shape[2], real_imgs.shape[3])).to(device)
                    
                    for j, sem_label in enumerate(semantic_choices):
                        g_mask_choice[j, 0] = g_mask[j, sem_label]
                        
                    
                    with torch.no_grad():
                        g_img_preds, g_img_positions_pred = discriminator_global_ddp(g_imgs[:,-3:], g_mask)
                        g_sem_img_preds, g_sem_mask_preds = discriminator_local_ddp(g_imgs[:,-3:], g_mask_choice)

                    #view loss
                    g_position_loss = torch.nn.SmoothL1Loss()(g_img_positions_pred, g_pos) * metadata['pos_lambda']
                    g_cross_entropy = torch.nn.CrossEntropyLoss()(g_sem_mask_preds, semantic_choices) #cross entropy

                    #eikonol and minimal surface loss
                    eikonol_loss, minimal_surface_loss = eikonol_surface_loss(g_grad_sdf, g_sdf, metadata['ms_beta'])
                    #TODO: where is eikonol loss
                    total_g_loss = torch.nn.functional.softplus(-g_img_preds).mean() + g_position_loss + metadata['eikonol_lambda']*eikonol_loss + \
                                   metadata['minimal_surface_lambda']*minimal_surface_loss + (torch.nn.functional.softplus(-g_sem_img_preds).mean() + g_cross_entropy)*metadata['local_d_lambda']


                    generator_losses.append(total_g_loss.item())

                scaler.scale(total_g_loss).backward()

            if rank == 0:
                logger.add_scalar('g_loss', total_g_loss.item(), generator_all.step)

            scaler.unscale_(optimizer_G)
            ##Caution: gad clips are not there in original paper, so use it with caution
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))             
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            #why there are 2 ema's?
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())

            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [Epoch: {discriminator_global.epoch}/{opt.n_epochs}] [D global loss: {d_global_loss.item()}] [D local loss: {d_local_loss.item()}] [G loss: {total_g_loss.item()}] [Step: {discriminator_global.step}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [Scale: {scaler.get_scale()}]")
                
                if discriminator_global.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            ##TODO: not sure about this change of std deviation
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            ## TODO: rectify the stage forward method. output should be n x (3 + k) x img_size x img_size
                            gen_imgs, gen_masks_out, gen_sigma = generator_ddp.module.stage_forward(z_sample_fixed.to(device), **copied_metadata)
                            gen_labels = mask2color(gen_masks_out)
                    

                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_global.step}_seg_fixed.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_global.step}_img_fixed.png"), nrow=5, normalize=True)
                
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            ##TODO: not sure about this change of std deviation
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            ## TODO: rectify the stage forward method. output should be n x (3 + k) x img_size x img_size
                            gen_imgs, gen_masks_out, gen_sigma = generator_ddp.module.stage_forward(z_sample_fixed.to(device), **copied_metadata)
                            gen_labels = mask2color(gen_masks_out)
                    

                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_global.step}_seg_tilted.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_global.step}_img_tilted.png"), nrow=5, normalize=True)

                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            ##TODO: not sure about this change of std deviation
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            ## TODO: rectify the stage forward method. output should be n x (3 + k) x img_size x img_size
                            gen_imgs, gen_masks_out, gen_sigma = generator_ddp.module.stage_forward(z_sample_fixed.to(device), **copied_metadata)
                            gen_labels = mask2color(gen_masks_out)
                    

                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_global.step}_seg_fixed_ema.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_global.step}_img_fixed_ema.png"), nrow=5, normalize=True)
                
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            ##TODO: not sure about this change of std deviation
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            ## TODO: rectify the stage forward method. output should be n x (3 + k) x img_size x img_size
                            gen_imgs, gen_masks_out, gen_sigma = generator_ddp.module.stage_forward(z_sample_fixed.to(device), **copied_metadata)
                            gen_labels = mask2color(gen_masks_out)
                    

                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_global.step}_seg_tilted_ema.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_global.step}_img_tilted_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            ##TODO: not sure about this change of std deviation
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            ##TODO: need to edit this too
                            # copied_metadata['psi'] = 0.7
                            ## TODO: rectify the stage forward method. output should be n x (3 + k) x img_size x img_size
                            gen_imgs, gen_masks_out, gen_sigma = generator_ddp.module.stage_forward(torch.randn_like(z_sample_fixed).to(device), **copied_metadata)
                            gen_labels = mask2color(gen_masks_out)
                    
                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_global.step}_seg_random.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_global.step}_img_random.png"), nrow=5, normalize=True)

                    ema.restore(generator_ddp.parameters())

                if discriminator_global.step % opt.sample_interval == 0:
                    torch.save(ema.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_ema.pth'))
                    torch.save(ema2.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_ema2.pth'))
                    torch.save(generator_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_generator_all.pth'))
                    torch.save(discriminator_global_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_discriminator_global.pth'))
                    torch.save(discriminator_local_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_discriminator_local.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_optimizer_G.pth'))
                    torch.save(optimizer_global_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_optimizer_global_D.pth'))
                    torch.save(optimizer_local_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_optimizer_local_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

            if opt.eval_freq > 0 and (discriminator_global.step + 1)%opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    ##TODO: we need to copy this fid evaluation
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, **metadata)
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()

                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, **metadata)
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator_global.step}:{fid}')
                    logger.add_scalar('fid', fid, discriminator_global.step)

                torch.cuda.empty_cache()

            discriminator_global.step += 1
            discriminator_local.step += 1
            generator_all.step += 1

        discriminator_global.epoch += 1
        discriminator_local.epoch += 1
        generator_all.epoch += 1

    #save the final snapshot of the model at the end of all epochs.
    torch.save(ema.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_ema.pth'))
    torch.save(ema2.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_ema2.pth'))
    torch.save(generator_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_generator_all.pth'))
    torch.save(discriminator_global_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_discriminator_global.pth'))
    torch.save(discriminator_local_ddp.module, os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_discriminator_local.pth'))
    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_optimizer_G.pth'))
    torch.save(optimizer_global_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_optimizer_global_D.pth'))
    torch.save(optimizer_local_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_optimizer_local_D.pth'))
    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, str(discriminator_global.step) + '_final_scaler.pth'))
    torch.save(generator_losses, os.path.join(opt.output_dir, '_finalgenerator.losses'))
    torch.save(discriminator_losses, os.path.join(opt.output_dir, '_final_discriminator.losses'))
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)
    parser.add_argument('--num_gpus', type=int, default=1)
    
    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = opt.num_gpus
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)   



            
            


            





                        




