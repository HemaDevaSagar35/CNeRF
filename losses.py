import math
import torch
from torch import autograd
from torch.nn import functional as F

def discriminator_loss_r1(r_img_preds, real_imgs, real_labels, scaler):
    grad_img_real, grad_mask_real = torch.autograd.grad(
                    outputs=scaler.scale(r_img_preds.sum()),
                    inputs=[real_imgs, real_labels],
                    create_graph=True
                )
    inv_scale = 1./scaler.get_scale()
    grad_img_real =  grad_img_real * inv_scale
    grad_mask_real = grad_mask_real * inv_scale

    with torch.cuda.amp.autocast():
        grad_img_penalty = (grad_img_real.view(grad_img_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_mask_penalty = (grad_mask_real.view(grad_mask_real.size(0), -1).norm(2, dim=1) ** 2).mean()

    return grad_img_penalty, grad_mask_penalty

def eikonol_surface_loss(grad_sdf, sdf, beta=100):
    # eikonol_loss = torch.tensor(0, device=grad_sdf.device)
    # minimal_surface_loss = 
    if grad_sdf is None:
        eikonol_loss = torch.tensor(0, device=grad_sdf.device)
    else:
        eikonol_loss = ((grad_sdf.norm(dim=-1) - 1)**2).mean()
    if sdf is None:
        minimal_surface_loss = torch.tensor(0, device=grad_sdf.device)
    else:
        minimal_surface_loss = torch.exp(-1.0*beta*torch.abs(grad_sdf)).mean()
    
    return eikonol_loss, minimal_surface_loss