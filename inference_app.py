import torch
import trimesh
import numpy as np
import skvideo.io
import imageio
from scipy.interpolate import CubicSpline
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms
from options import BaseOptions
from model import Generator
from utils import (
    generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
    xyz2mesh, create_cameras, create_mesh_renderer, add_textures,
    )
from pytorch3d.structures import Meshes
from dataset import color_segmap
from pdb import set_trace as st
import os

class InferenceObj():
    def __init__(self):
        self.device = "cuda"
        self.input = None

        self.opt = BaseOptions().parse()
        self.opt.model.is_test = True
        self.opt.model.style_dim = 256
        self.opt.model.freeze_renderer = False
        self.opt.rendering.depth = 3
        self.opt.rendering.width = 128
        self.opt.rendering.no_features_output = False
        self.opt.inference.size = self.opt.model.size
        self.opt.inference.camera = self.opt.camera
        self.opt.inference.renderer_output_size = self.opt.model.renderer_spatial_output_dim
        self.opt.inference.style_dim = self.opt.model.style_dim
        self.opt.inference.project_noise = self.opt.model.project_noise
        self.opt.rendering.perturb = 0
        self.opt.rendering.force_background = True
        self.opt.rendering.static_viewdirs = True
        self.opt.rendering.return_sdf = False
        self.opt.rendering.return_xyz = False
        self.opt.rendering.N_samples = 24
        # opt.experiment.ckpt = 25000
        self.opt.training.checkpoints_dir = 'checkpoints'
        self.opt.training.trained_ckpt = '../checkpoints/models_0025000.pt'
        # return device, opt

    def load_model(self):
        os.makedirs(self.opt.inference.results_dir, exist_ok=True)
        checkpoint_path = self.opt.training.trained_ckpt
        checkpoint = torch.load(checkpoint_path)

        self.g_ema = Generator(self.opt.model, self.opt.rendering, full_pipeline=False).to(self.device)
        pretrained_weights_dict = checkpoint["g_ema"]
        model_dict = self.g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v
        self.g_ema.load_state_dict(model_dict)
        with torch.no_grad():
            self.mean_latent = self.g_ema.mean_latent(self.opt.inference.truncation_mean, self.device)
        
        self.opt = self.opt.inference
        self.g_ema.eval()
        # return self.g_ema, self.mean_latent

    def random_input(self):
        self.input = torch.randn(1, self.opt.style_dim, device=self.device)
    
    def gen_styles(self):
        self.styles = self.g_ema.style(self.input)
    
    def gen_style_global(self, styles):
        styles = self.opt.truncation_ratio * styles + (1-self.opt.truncation_ratio) * self.mean_latent[0]
        self.styles_global = styles
    
    def camera_attributes(self, idx=5):
        num_frames = 1
        trajectory = np.zeros((num_frames,3), dtype=np.float32)

        t1 = np.linspace(-1.5, 1.5, 10)[idx]
        t2 = 0.8 * np.ones(num_frames)

        fov = self.opt.camera.fov
        elev = self.opt.camera.elev * t2
        azim = self.opt.camera.azim * t1

        trajectory[:num_frames,0] = azim
        trajectory[:num_frames,1] = elev
        trajectory[:num_frames,2] = fov

        trajectory = torch.from_numpy(trajectory).to(self.device)

        sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
        generate_camera_params(self.opt.renderer_output_size, self.device, locations=trajectory[:,:2],
                            fov_ang=trajectory[:,2:], dist_radius=self.opt.camera.dist_radius)

        cameras = create_cameras(azim=np.rad2deg(trajectory[0,0].cpu().numpy()),
                                elev=np.rad2deg(trajectory[0,1].cpu().numpy()),
                                dist=1, device=self.device)
        return sample_cam_extrinsics, sample_focals, sample_near, sample_far, cameras
    
    def output(self, idx=5, sem = None):
        sample_cam_extrinsics, sample_focals, sample_near, sample_far, cameras = self.camera_attributes(idx)
        styles_new = self.styles_global.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        out = self.g_ema([styles_new],
            sample_cam_extrinsics,
            sample_focals,
            sample_near,
            sample_far,
            truncation=self.opt.truncation_ratio,
            truncation_latent=self.mean_latent,
            input_is_latent=True,
            randomize_noise=False,
            project_noise=self.opt.project_noise,
            mesh_path=frontal_marching_cubes_mesh_filename if self.opt.project_noise else None,
            styles_global=[self.styles_global],
            semantics=sem
                            )

        _, img, seg, f, img_sem = out
        del out
        torch.cuda.empty_cache()

        img = img.clamp(-1,1) * 0.5 + 0.5
        img = img.detach().cpu()
        seg = color_segmap(seg)
        seg = seg.detach().cpu()
        seg = seg/255

        if img_sem is not None:
            img_sem = img_sem.permute(0,3,1,2).contiguous()
            img_sem = img_sem.clamp(-1, 1) * 0.5 + 0.5
            img_sem = img_sem.detach().cpu()

        # transforms.functional.to_pil_image(img[0])
        # transforms.functional.to_pil_image(seg[0])

            return transforms.functional.to_pil_image(img[0]), transforms.functional.to_pil_image(seg[0]), transforms.functional.to_pil_image(img_sem[0])

        return transforms.functional.to_pil_image(img[0]), transforms.functional.to_pil_image(seg[0]), None
        
    def cal_direction(self):
        with torch.no_grad():
            self.direction = self.styles/torch.norm(self.styles)

    def output_manipulate(self, sem_idx, lambda_val):
        print(torch.sum(self.styles))
        styles_change = self.styles + lambda_val*self.direction
        print(torch.sum(self.styles))
        print(torch.sum(styles_change))
        # styles_change = self.styles
        styles_change = self.opt.truncation_ratio * styles_change + (1 - self.opt.truncation_ratio)*self.mean_latent[0]
        
        sample_cam_extrinsics, sample_focals, sample_near, sample_far, cameras = self.camera_attributes()
        styles_new = self.styles_global.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        styles_new[:,2*sem_idx:(2*sem_idx + 1)] = styles_change

        out = self.g_ema([styles_new],
            sample_cam_extrinsics,
            sample_focals,
            sample_near,
            sample_far,
            truncation=self.opt.truncation_ratio,
            truncation_latent=self.mean_latent,
            input_is_latent=True,
            randomize_noise=False,
            project_noise=self.opt.project_noise,
            mesh_path=frontal_marching_cubes_mesh_filename if self.opt.project_noise else None,
            styles_global=[self.styles_global],
            semantics=None
                            )

        _, img, seg, f, img_sem = out
        del out
        torch.cuda.empty_cache()

        img = img.clamp(-1,1) * 0.5 + 0.5
        img = img.detach().cpu()
        seg = color_segmap(seg)
        seg = seg.detach().cpu()
        seg = seg/255

        
        return transforms.functional.to_pil_image(img[0]), transforms.functional.to_pil_image(seg[0])
       
            




