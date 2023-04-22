import math



def extract_metadata(curriculum, current_step):
    return_dict = {}
    
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

cnerf_stage_one = {
    'z_dim' : 256,
    'z_dist' : 'gaussian',
    'hidden_dim' : 128,
    'latent_dim' : 256,
    'semantic_classes' : 12,
    'output_dim' : 133,
    'blocks' : 3,
    'gen_lr' : 2e-5,
    'betas' : (0, 0.9),
    'weight_decay' : 0,
    'gen_d' : 2e-4,
    'disable_scaler' : False,
    'dataset' : 'FFHQDataset',
    'mixing_bool' : 0.9,
    'batch_split' : 4,
    'r1_img_lambda' : 10,
    'r1_mask_lambda' : 1000,
    'pos_lambda' : 15,
    'grad_clip' : 10,
    'local_d_lambda' : 1.0,
    'img_size' : 64,
    'ms_beta' : 100,
    'num_steps' : 24,
    'eikonol_lambda' : 0.1,
    'minimal_surface_lambda' : 0.001,
    'batch_size' : 4,
    'h_stddev' : 0.3,
    'v_stddev' : 0.155,
    'h_mean': 0,
    'v_mean': 0,
    'fov' : 12,
    'ray_start' : 0.88,
    'ray_end' : 1.12,
    'sample_dist' : 'gaussian',
    'max_batch' : 4,
    'freq_bias_init' : 30,
    'freq_std_init' : 15,
    'phase_bias_init' : 0,
    'phase_std_init' : 0.25,
    'dataset': 'FFHQDataset',
    'dataset_path': 'data/ffhq_mask',
   
}